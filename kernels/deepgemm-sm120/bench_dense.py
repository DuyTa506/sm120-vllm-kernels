"""Dense FP8 GEMM on the real Qwen3.6-35B-A3B shapes: DeepGEMM nv_dev (SM120) vs vLLM cutlass_scaled_mm vs cuBLASLt
(torch._scaled_mm rowwise). CUDA-graph timed, weights rotated through distinct copies so L2 (128 MB) cannot help.

Quantization contract differs: vLLM/cuBLASLt = per-token activation scale + per-output-channel weight scale
(one fp32 multiply in the epilogue). DeepGEMM = per-1x128 UE8M0 scale on A and per-1x128-row UE8M0 scale on B
(scale applied per K-block of 128 inside the mainloop). Same FLOPs and same bytes (+ small scale tensors), but a
different, more expensive scaling contract, so DG numbers are a proxy for kernel/tile quality, not a drop-in.
"""
import argparse
import re
import sys

import torch

import deep_gemm
from common import Out, bench_graph, calc_diff, capture_fd1, fp8_rand, gemm_kernel, gpu_state, make_act, pow2_sf

ap = argparse.ArgumentParser()
ap.add_argument("--shapes", default="gdn_qkvz,gdn_out,attn_qkv,attn_o,lm_head")
ap.add_argument("--ms", default="1,2,3,4,8,16,64,256,1024,8192")
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--out", default="dense.jsonl")
ap.add_argument("--warm-only", action="store_true", help="only run each case once (JIT compile), no timing")
ap.add_argument("--slice", default="0/1", help="i/n: process only every n-th case (parallel JIT warm-up)")
ap.add_argument("--no-vllm", action="store_true")
args = ap.parse_args()

SHAPES = {  # name -> (N, K, calls per forward)
    "gdn_qkvz": (12288, 2048, 30), "gdn_out": (2048, 4096, 30), "attn_qkv": (9216, 2048, 10),
    "attn_o": (2048, 4096, 10), "lm_head": (248320, 2048, 1),
}
PEAK, BW = 1100e12, 1.8e12
si, sn = map(int, args.slice.split("/"))
out = Out(args.out) if not args.warm_only else None
ops = None
if not args.no_vllm:
    from vllm import _custom_ops as ops  # noqa: E402

sk = None
if not args.no_vllm:
    try:
        sys.path.insert(0, "/work/ours_ref")
        from skinny.build import load as sk_load  # copy of our shipped skinny-M FP8 GEMV (kernels/dense-fp8-gemm)
        from skinny.dispatch import pick as skinny_pick
        sk = sk_load()
    except Exception as e:  # noqa: BLE001
        print("skinny unavailable", type(e).__name__, str(e)[:200], flush=True)
print("GPU", torch.cuda.get_device_name(0), "SMs", deep_gemm.get_num_sms(), "pdl default", deep_gemm.get_pdl(), "|", gpu_state(), flush=True)
case_i = -1
for shape in args.shapes.split(","):
    N, K, _ = SHAPES[shape]
    ms = [int(x) for x in args.ms.split(",")]
    cases = [M for M in ms if not (shape == "lm_head" and M > 256)]
    cases = [M for M in cases if (case_i := case_i + 1) % sn == si]
    if not cases:
        continue
    torch.manual_seed(1)
    # weight: fp8 bytes + per-row packed UE8M0 scale for DG; per-channel fp32 scale for vLLM / cuBLASLt (same bytes)
    w = torch.empty(N, K, dtype=torch.float8_e4m3fn, device="cuda")
    for r0 in range(0, N, 16384):
        w[r0:r0 + 16384] = fp8_rand((min(16384, N - r0), K), 1.0)
    w_sf = pow2_sf((N, K // 128), -9, -6)
    w_sf_packed = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(w_sf)
    w_ch = pow2_sf((N,), -9, -6)
    for M in cases:
        R = 2 if (M >= 4096) else max(2, -(-(640 << 20) // (N * K)))
        if shape == "lm_head":
            R = 2
        if args.warm_only:
            R = 1
        ws = [w] + [w.clone() for _ in range(R - 1)]
        sfs = [w_sf_packed] + [w_sf_packed.clone() for _ in range(R - 1)]
        x = make_act(M, K, seed=M)
        # --- DG operands: per-1x128 UE8M0 A scales
        xs = x.float().view(M, K // 128, 128)
        a_sf = torch.exp2(torch.ceil(torch.log2((xs.abs().amax(-1).clamp(1e-4) / 448.0))))
        a_q = (xs / a_sf.unsqueeze(-1)).view(M, K).to(torch.float8_e4m3fn)
        a_sf_packed = deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor(a_sf.contiguous())
        d = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")

        def dg(i, d=d):
            deep_gemm.fp8_fp4_gemm_nt((a_q, a_sf_packed), (ws[i], sfs[i]), d, recipe_a=(1, 128), recipe_b=(1, 128))

        print(f"CASE {shape} M={M} N={N} K={K} R={R}", flush=True)
        _, cfg = capture_fd1(lambda: dg(0))
        cfg_m = re.search(r"SM120GemmConfig\(layout=Layout\(swap_ab=(\d), block_m=(\d+), block_n=(\d+), block_k=(\d+).*?num_stages=(\d+).*?split_k=(\d+)", cfg)
        cfgd = dict(zip(("layout_swap", "block_m", "block_n", "block_k", "stages", "split_k"), map(int, cfg_m.groups()))) if cfg_m else {"raw": cfg[-300:]}
        cfgd["api_swap_ab"] = M <= 16
        cfgd["kernel"] = gemm_kernel(lambda: dg(0))
        if args.warm_only:
            continue
        # correctness vs fp32 reference of the quantized operands
        ref = None
        if M * N <= (1 << 28):
            a_deq = a_q.float() * a_sf.repeat_interleave(128, 1)
            b_deq = w.float() * w_sf.repeat_interleave(128, 1)
            ref = a_deq @ b_deq.T
            dg(0)
            cfgd["calc_diff_vs_fp32_ref"] = calc_diff(d.float(), ref)
        flops = 2 * M * N * K
        byt = N * K + M * K + 2 * M * N
        reps = max(1, 8 // R) if M < 4096 else 2

        def rec(impl, us, lo, hi, extra=None):
            tf, gb = flops / (us * 1e-6) / 1e12, byt / (us * 1e-6) / 1e9
            row = dict(shape=shape, M=M, N=N, K=K, impl=impl, us=us, us_min=lo, us_max=hi, tflops=tf, gbps=gb,
                       pct_peak=100 * tf * 1e12 / PEAK, pct_bw=100 * gb * 1e9 / BW, gpu=gpu_state(), **(extra or {}))
            out.write(**row)
            print(f"  {impl:10s} {us:9.1f} us {tf:7.1f} TFLOPS {gb:7.0f} GB/s", flush=True)

        us = bench_graph([lambda i=i: dg(i) for i in range(R)] * reps, iters=args.iters)
        rec("dg", *us, extra=cfgd)
        old_pdl = deep_gemm.get_pdl()
        deep_gemm.set_pdl(True)
        try:
            us = bench_graph([lambda i=i: dg(i) for i in range(R)] * reps, iters=args.iters)
            rec("dg_pdl", *us, extra=cfgd)
        finally:
            deep_gemm.set_pdl(old_pdl)
        if ops is not None:
            from vllm import _custom_ops as vops
            a_v, a_vs = vops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
            outv = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
            try:
                fv = lambda i: vops.cutlass_scaled_mm(a_v, ws[i].t(), a_vs, w_ch, torch.bfloat16)
                fv(0)
                rec("vllm", *bench_graph([lambda i=i: fv(i) for i in range(R)] * reps, iters=args.iters))
            except Exception as e:  # noqa: BLE001
                print("  vllm FAILED", type(e).__name__, str(e)[:100], flush=True)
            try:
                ft = lambda i: torch._scaled_mm(a_v, ws[i].t(), scale_a=a_vs, scale_b=w_ch.view(1, -1), out_dtype=torch.bfloat16)
                ft(0)
                rec("cublaslt", *bench_graph([lambda i=i: ft(i) for i in range(R)] * reps, iters=args.iters))
            except Exception as e:  # noqa: BLE001
                print("  cublaslt FAILED", type(e).__name__, str(e)[:100], flush=True)
        if ops is not None and sk is not None and M <= 64:
            pk = skinny_pick(M, N, K)
            if pk:
                kw, kb = pk
                outs = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
                fk = lambda i: sk.mm(outs, a_v, ws[i], a_vs.reshape(-1), w_ch, kw, kb)
                fk(0)
                rec("skinny", *bench_graph([lambda i=i: fk(i) for i in range(R)] * reps, iters=args.iters), extra=dict(kw=kw, kb=kb))
        del ws, sfs, d, ref
        torch.cuda.empty_cache()
    del w
    torch.cuda.empty_cache()
print("done", gpu_state(), flush=True)
