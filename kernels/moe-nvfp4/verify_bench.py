"""VERIFIER bench: FlashInfer CUTLASS (vLLM default) vs b12x with fixed scales, as the hybrid patch calls it.

Differences from bench_b12x_vs_cutlass.py:
  * b12x side includes what FlashInferB12xExperts.apply does in the patched vLLM: negative-id masking
    (compare + 2 masked_fill), topk_ids.to(int32), and output.copy_(wrapper_output) into the MoE output.
    The "b12x_raw" column is the author's measurement (wrapper.run only).
  * CUTLASS and b12x graphs are replayed interleaved (A,B,A,B...) so clock/thermal drift hits both sides.
  * several layers (routing skew differs per layer) and a balanced uniform-random routing case.
  * persistent memory of one B12xMoEWrapper(use_cuda_graph=True, max_num_tokens=16384) as vLLM creates it.
usage: verify_bench.py [layer ...]
"""
import sys, statistics, torch
from moe_common import *


def capture(fn, inner=5):
    fn(); torch.cuda.synchronize()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(inner):
            fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    return g


def time_interleaved(graphs, reps=40, inner=5):
    ts = [[] for _ in graphs]
    for _ in range(reps):
        for i, g in enumerate(graphs):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); g.replay(); b.record(); b.synchronize()
            ts[i].append(a.elapsed_time(b) * 1e3 / inner)
    return [statistics.median(t) for t in ts], [min(t) for t in ts]


def balanced_routing(M, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.argsort(torch.rand(M, E, device="cuda", generator=g), dim=1)[:, :TOPK].to(torch.int32)
    w = torch.rand(M, TOPK, device="cuda", generator=g)
    return (w / w.sum(-1, keepdim=True)).float(), ids.contiguous()


def main():
    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe import B12xMoEWrapper
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    layers = [int(a) for a in sys.argv[1:]] or [19, 0, 39]
    for L in layers:
        raw = load_layer_raw(L)
        prep = prep_cutlass(raw)
        xall = load_hidden(L)
        s13m = convert_sf_to_mma_layout(prep["s13"].reshape(E * 2 * I, H // 16), m=2 * I, k=H, num_groups=E)
        s2m = convert_sf_to_mma_layout(prep["s2"].reshape(E * H, I // 16), m=H, k=I, num_groups=E)
        a13v = (1.0 / prep["a1_gscale"]).contiguous(); a2v = (1.0 / prep["a2_gscale"]).contiguous()
        w1a = (1.0 / raw["wgs13"]).contiguous(); w2a = ((1.0 / raw["wgs2"]) * a2v).contiguous()
        torch.cuda.synchronize(); m0 = torch.cuda.memory_allocated()
        wr = B12xMoEWrapper(num_experts=E, top_k=TOPK, hidden_size=H, intermediate_size=I,
                            use_cuda_graph=True, max_num_tokens=16384, activation="silu")
        torch.cuda.synchronize(); m1 = torch.cuda.memory_allocated()
        mma_sf_bytes = s13m.numel() * s13m.element_size() + s2m.numel() * s2m.element_size()
        print(f"== layer {L}: B12xMoEWrapper(max_num_tokens=16384) persistent buffers {(m1 - m0) / 2**20:.1f} MiB/layer"
              f" (+ MMA-layout scale copies {mma_sf_bytes / 2**20:.1f} MiB/layer); x40 layers = "
              f"{40 * (m1 - m0 + mma_sf_bytes) / 2**30:.2f} GiB", flush=True)
        print(f"{'M':>6} {'routing':>8} {'cutlass':>9} {'b12x_int':>9} {'b12x_raw':>9} {'int/cut':>8} {'raw/cut':>8}"
              f" {'top1 expert rows':>16}", flush=True)
        cases = [(M, "real") for M in (1024, 2048, 4096, 8192, 16384)] + [(8192, "balanced")]
        for M, kind in cases:
            if M <= xall.shape[0]:
                x = xall[:M].contiguous()
            else:
                x = xall[torch.randint(0, xall.shape[0], (M,), device="cuda",
                                       generator=torch.Generator(device="cuda").manual_seed(1))].contiguous()
            wts, ids = route(x, raw["gate"]) if kind == "real" else balanced_routing(M)
            top1 = torch.bincount(ids.view(-1).long(), minlength=E).max().item()
            out_c = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
            out_b = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
            with autotune(True):
                run_cutlass(prep, x, ids, wts, out=out_c)
            torch.cuda.synchronize()

            def f_cut():
                run_cutlass(prep, x, ids.to(torch.int), wts, out=out_c)

            def f_b12x_int():
                skip = ids < 0
                ids2 = ids.masked_fill(skip, 0)
                wts2 = wts.masked_fill(skip, 0.0)
                y = wr.run(x=x, w1_weight=prep["w13"], w1_weight_sf=s13m, w1_alpha=w1a, fc2_input_scale=a2v,
                           w2_weight=prep["w2"], w2_weight_sf=s2m, w2_alpha=w2a,
                           token_selected_experts=ids2.to(torch.int32), token_final_scales=wts2,
                           input_global_scale=a13v)
                out_b.copy_(y)

            def f_b12x_raw():
                wr.run(x=x, w1_weight=prep["w13"], w1_weight_sf=s13m, w1_alpha=w1a, fc2_input_scale=a2v,
                       w2_weight=prep["w2"], w2_weight_sf=s2m, w2_alpha=w2a,
                       token_selected_experts=ids, token_final_scales=wts, input_global_scale=a13v)

            gs = [capture(f) for f in (f_cut, f_b12x_int, f_b12x_raw)]
            med1, _ = time_interleaved(gs)
            med2, _ = time_interleaved(gs[::-1])   # reversed order, second pass
            med2 = med2[::-1]
            tc, ti, tr = [(a + b) / 2 for a, b in zip(med1, med2)]
            rel = rel_err(out_b, out_c)["rel_l2"]
            print(f"{M:6d} {kind:>8} {tc:9.1f} {ti:9.1f} {tr:9.1f} {ti / tc:8.3f} {tr / tc:8.3f} {top1:16d}"
                  f"   (pass1 {med1[0]:.1f}/{med1[1]:.1f}/{med1[2]:.1f}, pass2 {med2[0]:.1f}/{med2[1]:.1f}/{med2[2]:.1f};"
                  f" rel_l2 b12x vs cut {rel:.4f})", flush=True)
            del gs
        del wr, raw, prep, s13m, s2m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
