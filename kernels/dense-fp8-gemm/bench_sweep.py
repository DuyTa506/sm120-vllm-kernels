"""Sweep vLLM cutlass_scaled_mm against alternative kernels on the real dense FP8 shapes.
For every (shape, M, impl): us/call (CUDA graph, weights rotated so they are read cold), achieved
TFLOPS / GB/s, and numerics vs vLLM's output (bitwise-equal fraction, max abs diff) and vs an fp32
reference (max abs / max rel error)."""
import argparse
import sys

import torch

sys.path.insert(0, "/work/kernels/dense-fp8-gemm")
from common import SHAPES, bench_graph, load_weight, make_act, quant_per_token  # noqa: E402

from vllm import _custom_ops as ops  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shapes", default="gdn_qkvz,gdn_out,attn_qkv,attn_o,lm_head")
ap.add_argument("--ms", default="1,3,4,12,16,48,64,192,8192,16384")
ap.add_argument("--impls", default="auto")
ap.add_argument("--iters", type=int, default=30)
args = ap.parse_args()

import cutlass_ext.build as cb  # noqa: E402
import skinny.build as sbld  # noqa: E402

CX = cb.load()
SK = sbld.load()
PEAK, BW = 1100e12, 1.8e12


def make_impl(impl, a, a_s, w, w_s):
    M, N = a.shape[0], w.shape[0]
    wt = w.t()
    if impl == "vllm":
        return lambda: ops.cutlass_scaled_mm(a, wt, a_s, w_s, torch.bfloat16)
    if impl == "torch":
        sb = w_s.view(1, -1)
        return lambda: torch._scaled_mm(a, wt, scale_a=a_s, scale_b=sb, out_dtype=torch.bfloat16)
    if impl.startswith("skinny"):  # skinny_kw4_kb4
        _, kw, kb = impl.split("_")
        kw, kb = int(kw[2:]), int(kb[2:])
        out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        sa = a_s.reshape(-1)

        def f():
            SK.mm(out, a, w, sa, w_s, kw, kb)
            return out
        return f
    if impl.startswith("cx:"):  # cx:<config>[:raster:swizzle:splits:decomp]
        parts = impl.split(":")
        name = parts[1]
        vals = [int(x) for x in parts[2:]]
        raster, swz, splits, decomp = vals + [0, 1, 1, 0][len(vals):]
        out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        sa = a_s.reshape(-1).contiguous()

        def f():
            CX.run(name, out, a, wt, sa, w_s, raster, swz, splits, decomp)
            return out
        return f
    raise ValueError(impl)


def default_impls(M, K):
    if M <= 64:
        L = ["vllm", "torch", "cx:v_16x64x128_pp", "cx:v_32x64x128_pp", "cx:v_64x64x128_pp",
             "cx:16x128x128_pp", "cx:16x32x128_pp"]
        for kw, kb in [(1, 4), (2, 4), (4, 4), (4, 2), (8, 1), (8, 2), (8, 4), (16, 1), (16, 2)]:
            if K % (kw * 128 * kb) == 0:
                L.append(f"skinny_kw{kw}_kb{kb}")
        for sp in (2, 4, 8):
            L.append(f"cx:sk_128x64x128_coop:0:1:{sp}:2")
        return L
    if M <= 256:
        return ["vllm", "torch", "cx:v_64x64x128_pp", "cx:64x128x128_pp", "cx:128x64x128_pp",
                "cx:v_128x128x128_coop", "cx:sk_128x64x128_coop:0:1:2:2", "cx:sk_128x128x128_coop:0:1:2:2"]
    return ["vllm", "torch", "cx:v_128x128x128_coop", "cx:128x128x64_coop", "cx:128x256x64_coop",
            "cx:256x128x64_coop", "cx:128x128x128_pp", "cx:128x128x64_pp", "cx:64x128x128_pp",
            "cx:128x64x128_pp", "cx:64x256x64_pp", "cx:v_128x128x128_coop:1:1", "cx:v_128x128x128_coop:2:1",
            "cx:v_128x128x128_coop:0:2", "cx:v_128x128x128_coop:0:4"]


print(f"{'shape':9s} {'M':>6s} {'impl':34s} {'us':>9s} {'TFLOPS':>7s} {'%pk':>5s} {'GB/s':>6s} {'%bw':>5s} "
      f"{'bitEq%':>7s} {'maxd_vllm':>10s} {'maxabs_ref':>10s} {'rel_ref':>9s}")
for sn in args.shapes.split(","):
    N, K, _ = SHAPES[sn]
    w0, s0 = load_weight(sn, 0 if sn.startswith("gdn") else (3 if sn.startswith("attn") else None))
    for M in [int(x) for x in args.ms.split(",")]:
        if sn == "lm_head" and M > 256:
            continue
        R = 2 if M >= 4096 else max(1, min(96, -(-(640 << 20) // (N * K))))  # >640 MB rotation: cold weights
        ws = [w0] + [w0.clone() for _ in range(R - 1)]
        x = make_act(M, K, seed=M)
        a, a_s = quant_per_token(x)
        base = ops.cutlass_scaled_mm(a, w0.t(), a_s, s0, torch.bfloat16)
        ref = None
        if M * N <= (1 << 28):
            ref = (a.float() @ w0.float().t()) * a_s.view(-1, 1) * s0.view(1, -1)
        impls = default_impls(M, K) if args.impls == "auto" else args.impls.split(",")
        for impl in impls:
            try:
                f0 = make_impl(impl, a, a_s, w0, s0)
                o = f0().clone()
                torch.cuda.synchronize()
                eq = (o.view(torch.int16) == base.view(torch.int16)).float().mean().item() * 100
                md = (o.float() - base.float()).abs().max().item()
                if ref is not None:
                    d = (o.float() - ref).abs().max().item()
                    rel = d / ref.abs().max().item()
                else:
                    d = rel = float("nan")
                fns = [make_impl(impl, a, a_s, w, s0) for w in ws]
                reps = max(1, 8 // R) if M < 4096 else 2
                us = bench_graph(fns * reps, iters=args.iters)
            except Exception as e:  # noqa: BLE001
                print(f"{sn:9s} {M:6d} {impl:34s} FAILED {type(e).__name__}: {str(e)[:100]}", flush=True)
                continue
            fl = 2 * M * N * K / (us * 1e-6)
            by = (N * K + M * K + 2 * M * N) / (us * 1e-6)
            print(f"{sn:9s} {M:6d} {impl:34s} {us:9.1f} {fl/1e12:7.1f} {100*fl/PEAK:5.1f} {by/1e9:6.0f} "
                  f"{100*by/BW:5.1f} {eq:7.2f} {md:10.3g} {d:10.3g} {rel:9.2e}", flush=True)
        del ws, ref
        torch.cuda.empty_cache()
