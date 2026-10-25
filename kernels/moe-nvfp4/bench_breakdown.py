"""Per-kernel breakdown of one MoE layer call (router GEMM + top-k + quant + FlashInfer CUTLASS fused MoE),
real layer weights and captured real inputs, via torch.profiler (CUPTI kernel durations, eager launches,
median over 20 calls). Also times a dense NVFP4 GEMM of the same total work as a ceiling reference, and
the BF16 router GEMM (2048->256) alternatives.
usage: bench_breakdown.py [layer] [M ...]
"""
import sys, statistics, collections, torch
from moe_common import *
from bench_tactics import graph_time


def short(n):
    for k, v in [("expandInputRows", "expandInputRows"), ("doActivation", "doActivation"),
                 ("GroupProblemShape", "grouped_gemm"), ("blockExpertPrefixSum", "blockExpertPrefixSum"),
                 ("globalExpertPrefixSum", "globalExpertPrefixSum"), ("mergeExpertPrefixSum", "mergeExpertPrefixSum"),
                 ("fusedBuildExpertMaps", "buildExpertMaps"), ("finalizeMoeRouting", "finalizeMoeRouting"),
                 ("computeStrides", "computeStridesTmaWS"), ("cvt_fp16_to_fp4", "nvfp4_quant(vllm)"),
                 ("gemm", "router_gemm(cuBLAS)"), ("softmax", "softmax"), ("topk", "topk")]:
        if k in n:
            return v
    return n[:60]


def profile(fn, n=20):
    from torch.profiler import profile as prof, ProfilerActivity
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with prof(activities=[ProfilerActivity.CUDA]) as p:
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    acc = collections.defaultdict(list)
    for ev in p.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA and ev.device_time_total > 0:
            acc[ev.name].append(ev.device_time_total)
    rows = []
    for name, ts in acc.items():
        per_call = len(ts) / n
        rows.append((short(name), per_call, statistics.median(ts), name))
    return rows


def main():
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 19
    Ms = [int(a) for a in sys.argv[2:]] or [8192, 4096, 12]
    from flashinfer.autotuner import autotune
    raw = load_layer_raw(L)
    prep = prep_cutlass(raw)
    xall = load_hidden(L)
    if xall is None:
        xall = (torch.randn(8192, H, device="cuda") * 0.6).to(torch.bfloat16)
    for M in Ms:
        x = xall[:M].contiguous() if M <= xall.shape[0] else xall[torch.randint(0, xall.shape[0], (M,), device="cuda")].contiguous()
        wts, ids = route(x, raw["gate"])
        out = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
        with autotune(True):
            run_cutlass(prep, x, ids, wts, out=out)
        torch.cuda.synchronize()
        rows = profile(lambda: run_cutlass(prep, x, ids, wts, out=out))
        tot = sum(r[1] * r[2] for r in rows)
        flops = 2 * M * TOPK * (2 * I * H + H * I)
        print(f"\n== layer {L}, M={M} tokens: CUTLASS fused MoE kernels (eager, per call)  total {tot:.1f} us")
        for s, pc, med, full in sorted(rows, key=lambda r: -r[1] * r[2]):
            extra = ""
            if s == "grouped_gemm":
                extra = "  " + full[full.find("Shape<cute::C<") : full.find("Shape<cute::C<") + 70] if "Shape<cute::C<" in full else ""
            print(f"   {s:24s} x{pc:4.1f}  {med:8.1f} us{extra}")
        gg = [r for r in rows if r[0] == "grouped_gemm"]
        gsum = sum(r[1] * r[2] for r in gg)
        print(f"   grouped GEMMs total {gsum:.1f} us -> {flops / gsum / 1e6:.0f} TFLOPS (NVFP4 peak ~2200)")
        # router GEMM variants (bf16 2048->256), graph-timed
        g = raw["gate"]
        t_lin = graph_time(lambda: torch.nn.functional.linear(x, g))
        g257 = torch.cat([g, raw["shared_gate"]], 0).contiguous()
        g264 = torch.cat([g257, torch.zeros(7, H, device="cuda", dtype=g.dtype)], 0).contiguous()
        t_sg = graph_time(lambda: torch.nn.functional.linear(x, raw["shared_gate"]))
        t_264 = graph_time(lambda: torch.nn.functional.linear(x, g264))
        print(f"   router: F.linear 2048->256 {t_lin:.1f} us + shared_expert_gate 2048->1 {t_sg:.1f} us; "
              f"fused 2048->264 {t_264:.1f} us; ideal(max(bytes/1.8TB/s, flops/550T)) "
              f"{max(M*H*2/1.8e6, 2*M*H*256/550e6):.1f} us")


if __name__ == "__main__":
    main()
