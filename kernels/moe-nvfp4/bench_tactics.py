"""Sweep FlashInfer CUTLASS fused-MoE GEMM tactics on SM120 with a real layer + real routing.

Tactic space (FlashInfer 0.7.0.post1, SM120 NVFP4 grouped GEMM, see README):
  GEMM1 ids 0..19  = 10 CTA tiles x {normal, swap_ab}
  GEMM2 ids 20..59 = 10 CTA tiles x {plain epilogue, fused finalize} x {normal, swap_ab}
For every token count M the script (1) asks the FlashInfer autotuner (same as vLLM's warmup) which
tactics it picks, (2) times every valid GEMM1 tactic with GEMM2 fixed to the autotuned one and vice versa,
whole fused-MoE call, CUDA-graph replay, median of N, and (3) re-times the best pair vs the autotuned pair.
usage: bench_tactics.py [layer] [M ...]
"""
import sys, statistics, torch
from moe_common import *

TILES = ["128x128x128B", "128x128x64B", "256x128x64B", "128x256x64B", "128x128x256B", "256x128x128B",
         "128x32x128B", "128x32x64B", "128x64x128B", "128x64x64B"]


def name(t):
    if t < 20:
        return f"g1:{TILES[t % 10]}{'+swapAB' if t >= 10 else ''}"
    u = t - 20
    sw = u >= 20
    u %= 20
    return f"g2:{TILES[u % 10]}{'+finalize' if u >= 10 else ''}{'+swapAB' if sw else ''}"


def graph_time(fn, reps=30, inner=5):
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
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / inner)
    del g
    return statistics.median(ts)


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 19
    Ms = [int(a) for a in sys.argv[2:]] or [8192, 4096, 1024, 192, 48, 12, 3]
    torch.manual_seed(0)
    raw = load_layer_raw(L)
    prep = prep_cutlass(raw)
    xall = load_hidden(L)
    src = "captured" if xall is not None else "synthetic"
    if xall is None:
        xall = (torch.randn(8192, H, device="cuda") * 0.6).to(torch.bfloat16)
    from flashinfer.autotuner import autotune, AutoTuner
    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module
    print(f"layer {L}, {src} inputs; tactic sweep, whole fused-MoE call (quantized input precomputed), "
          f"CUDA graph, median us")
    for M in Ms:
        if M <= xall.shape[0]:
            x = xall[:M].contiguous() if M >= 1024 else xall[torch.randperm(xall.shape[0], device="cuda")[:M]].contiguous()
        else:
            x = xall[torch.randint(0, xall.shape[0], (M,), device="cuda")].contiguous()
        wts, ids = route(x, raw["gate"])
        xq = quant_input(x, prep["a1_gscale"])
        out = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
        cnt = torch.bincount(ids.view(-1).long(), minlength=E)
        # autotuner choice (what vLLM's warmup does), fresh tuner per M bucket
        AutoTuner.get().clear_cache() if hasattr(AutoTuner.get(), "clear_cache") else None
        with autotune(True):
            run_cutlass(prep, x, ids, wts, out=out, xq=xq)
        torch.cuda.synchronize()
        runner = list(get_cutlass_fused_moe_module("120").MoERunner.runner_dict.values())[0]
        n1, n2 = runner.get_gemm1_tactic_count(), runner.get_gemm2_tactic_count()
        valid1 = [t for t in range(n1) if runner.get_tactic_occupancy(t) > 0]
        valid2 = [t for t in range(n1, n1 + n2) if runner.get_tactic_occupancy(t) > 0]
        # read the tuned choice back from the tuner cache
        tuned = {}
        for k, v in AutoTuner.get().profiling_cache.items():
            if "fused_moe" in str(k.custom_op):
                tuned[str(k.custom_op).split("::")[-1] + str(k.nearest_profile[0])] = str(v)[:40]
        print(f"\n[autotuner pick for M={M}] {tuned}")
        t_auto = graph_time(lambda: run_cutlass(prep, x, ids, wts, out=out, xq=xq))
        flops = 2 * M * TOPK * (2 * I * H + H * I)
        print(f"\nM={M} tokens ({M*TOPK} routed rows; rows/expert mean {cnt.float().mean():.1f} max {cnt.max().item()} "
              f"active {int((cnt>0).sum())}): autotuned call {t_auto:8.1f} us = {flops/t_auto/1e6:6.0f} TFLOPS eff")
        # default pair to hold fixed: the best from a coarse search starting at autotuned (-1 means default)
        best2 = None
        res1 = {}
        for t in valid1:
            try:
                res1[t] = graph_time(lambda: run_cutlass(prep, x, ids, wts, out=out, xq=xq, profile_ids=[t, -1]))
            except Exception as e:
                res1[t] = float("nan")
        b1 = min(res1, key=lambda t: res1[t])
        res2 = {}
        for t in valid2:
            try:
                res2[t] = graph_time(lambda: run_cutlass(prep, x, ids, wts, out=out, xq=xq, profile_ids=[b1, t]))
            except Exception as e:
                res2[t] = float("nan")
        b2 = min(res2, key=lambda t: res2[t])
        for t in sorted(res1, key=lambda t: res1[t])[:6]:
            print(f"   {t:2d} {name(t):34s} {res1[t]:8.1f} us (gemm2 default)")
        for t in sorted(res2, key=lambda t: res2[t])[:6]:
            print(f"   {t:2d} {name(t):34s} {res2[t]:8.1f} us (gemm1={b1})")
        t_best = graph_time(lambda: run_cutlass(prep, x, ids, wts, out=out, xq=xq, profile_ids=[b1, b2]))
        t_auto2 = graph_time(lambda: run_cutlass(prep, x, ids, wts, out=out, xq=xq))
        print(f"  best pair [{b1},{b2}] {t_best:8.1f} us vs autotuned {t_auto2:8.1f} us "
              f"({(t_auto2-t_best)/t_auto2*100:+.1f}%)")


if __name__ == "__main__":
    main()
