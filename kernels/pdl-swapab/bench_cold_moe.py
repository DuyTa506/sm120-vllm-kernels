"""MoE decode with COLD weights: FlashInfer CUTLASS (vLLM default, swap-AB 128x32 tiles) vs FlashInfer b12x
(tokens on the MMA M side, CUDA-core direct-micro kernel for tiny batches), real layers, real routing.

The existing kernels/moe-nvfp4 benches replay one layer, so its ~41 MB of active-expert weights stay in L2
(128 MB). In a real step 40 layers stream through, so every layer reads its experts from DRAM. Here several
layers are called in one CUDA graph with a 300 MB memset between calls to flush L2; the flush-only graph is
subtracted. b12x runs with unit activation scales (the only setting that is correct for every M, see
kernels/moe-nvfp4 README 4c); correctness is printed as rel_l2 vs the CUTLASS output.
usage (inside the vLLM image, cwd anywhere):  bench_cold_moe.py [M ...]
"""
import statistics, sys
import torch
sys.path.insert(0, "/work/kernels/moe-nvfp4")
from moe_common import *  # noqa

LAYERS = [0, 9, 19, 29, 39]
Ms = [int(a) for a in sys.argv[1:]] or [3, 6, 12, 24, 48]


def graph_med(fn, reps=30):
    fn(); torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    return statistics.median(ts), min(ts)


def main():
    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe import B12xMoEWrapper
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    flush = torch.empty(300 << 20, dtype=torch.uint8, device="cuda")
    ls = []
    for L in LAYERS:
        raw = load_layer_raw(L)
        prep = prep_cutlass(raw)
        xall = load_hidden(L)
        s13m = convert_sf_to_mma_layout(prep["s13"].reshape(E * 2 * I, H // 16), m=2 * I, k=H, num_groups=E)
        s2m = convert_sf_to_mma_layout(prep["s2"].reshape(E * H, I // 16), m=H, k=I, num_groups=E)
        ones = torch.ones(E, device="cuda")
        d = dict(L=L, raw_gate=raw["gate"], prep=prep, xall=xall, s13m=s13m, s2m=s2m, ones=ones,
                 inv13=(1.0 / raw["wgs13"]).contiguous(), inv2=(1.0 / raw["wgs2"]).contiguous())
        d["wr"] = B12xMoEWrapper(num_experts=E, top_k=TOPK, hidden_size=H, intermediate_size=I,
                                 use_cuda_graph=True, max_num_tokens=max(Ms), activation="silu")
        del raw
        ls.append(d)
        torch.cuda.empty_cache()
    print(f"loaded layers {LAYERS}; GPU mem {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)

    def mk(d, M):
        # M tokens = M/3 independent windows of 3 consecutive real tokens (what an MTP-2 verify step feeds)
        n = d["xall"].shape[0] - 8
        x = torch.cat([d["xall"][(1000 + 977 * j) % n: (1000 + 977 * j) % n + min(3, M - 3 * j)] for j in range((M + 2) // 3)]).contiguous()
        wts, ids = route(x, d["raw_gate"])
        return x, wts, ids

    print(f"{'M':>4} {'distinct experts/layer':>24} {'cutlass us/layer':>17} {'b12x us/layer':>14} {'ratio':>6}  rel_l2(b12x vs cutlass) per layer")
    for M in Ms:
        calls_c, calls_b, rel, nd = [], [], [], []
        for d in ls:
            x, wts, ids = mk(d, M)
            out_c = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
            with autotune(True):
                run_cutlass(d["prep"], x, ids, wts, out=out_c)
            torch.cuda.synchronize()
            fc = (lambda d=d, x=x, ids=ids, wts=wts, out_c=out_c: run_cutlass(d["prep"], x, ids, wts, out=out_c))
            fb = (lambda d=d, x=x, ids=ids, wts=wts: d["wr"].run(
                x=x, w1_weight=d["prep"]["w13"], w1_weight_sf=d["s13m"], w1_alpha=d["inv13"], fc2_input_scale=d["ones"],
                w2_weight=d["prep"]["w2"], w2_weight_sf=d["s2m"], w2_alpha=d["inv2"], token_selected_experts=ids,
                token_final_scales=wts, input_global_scale=d["ones"]))
            yc = fc().clone(); yb = fb().clone()
            rel.append(rel_err(yb, yc)["rel_l2"]); nd.append(int(torch.unique(ids).numel()))
            calls_c.append(fc); calls_b.append(fb)

        def run_all(calls, with_flush=True):
            def f():
                for c in calls:
                    if with_flush:
                        flush.zero_()
                    if c is not None:
                        c()
            return f
        base, _ = graph_med(run_all([None] * len(ls)))
        tc, tc_min = graph_med(run_all(calls_c)); tb, tb_min = graph_med(run_all(calls_b))
        n = len(ls)
        cc, bb = (tc - base) / n, (tb - base) / n
        print(f"{M:4d} {sum(nd)/n:24.1f} {cc:17.1f} {bb:14.1f} {bb/cc:6.2f}  " + " ".join(f"{r:.3f}" for r in rel)
              + f"   (flush-only {base/n:.1f} us/layer; min-based c {(tc_min-base)/n:.1f} b {(tb_min-base)/n:.1f})", flush=True)


if __name__ == "__main__":
    main()
