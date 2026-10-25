"""Baseline sweep: vLLM cutlass_scaled_mm vs torch._scaled_mm (cuBLASLt rowwise) vs vLLM triton_scaled_mm
on the exact dense FP8 shapes of Qwen3.6-35B-A3B, per-token act scale + per-channel weight scale.
Timing: one CUDA graph holding a rotation over R distinct weight copies (total > L2) so weights are
read cold like in a real forward. Median of 30 replays, us per call."""
import argparse
import sys

import torch

sys.path.insert(0, "/work/kernels/dense-fp8-gemm")
from common import SHAPES, bench_graph, load_weight, make_act, quant_per_token  # noqa: E402

from vllm import _custom_ops as ops  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shapes", default=",".join(SHAPES))
ap.add_argument("--ms", default="1,3,4,12,16,48,64,192,8192,16384")
ap.add_argument("--impls", default="vllm,torch")
args = ap.parse_args()

PEAK = 1100e12
BW = 1.8e12


def impl_fns(name, a, a_s, w, w_s):
    wt = w.t()
    if name == "vllm":
        return lambda: ops.cutlass_scaled_mm(a, wt, a_s, w_s, torch.bfloat16)
    if name == "torch":
        sb = w_s.view(1, -1)
        return lambda: torch._scaled_mm(a, wt, scale_a=a_s, scale_b=sb, out_dtype=torch.bfloat16)
    if name == "triton":
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import triton_scaled_mm
        sb = w_s.view(-1, 1)
        return lambda: triton_scaled_mm(a, wt, a_s, sb, torch.bfloat16)
    raise ValueError(name)


print(f"{'shape':9s} {'M':>6s} {'impl':7s} {'us':>9s} {'TFLOPS':>7s} {'%pk':>5s} {'GB/s':>7s} {'%bw':>5s}")
for sn in args.shapes.split(","):
    N, K, _ = SHAPES[sn]
    w0, s0 = load_weight(sn, 0 if sn.startswith("gdn") else (3 if sn.startswith("attn") else None))
    for M in [int(x) for x in args.ms.split(",")]:
        R = max(1, min(16, (320 << 20) // (N * K)))  # >320 MB of weights in rotation
        if M >= 4096:
            R = 2
        ws = [w0] + [w0.clone() for _ in range(R - 1)]
        x = make_act(M, K)
        a, a_s = quant_per_token(x)
        for impl in args.impls.split(","):
            try:
                fns = [impl_fns(impl, a, a_s, w, s0) for w in ws]
                reps = max(1, 4 // R) if M < 4096 else 2
                us = bench_graph(fns * reps)
            except Exception as e:  # noqa: BLE001
                print(f"{sn:9s} {M:6d} {impl:7s} FAILED {type(e).__name__}: {str(e)[:120]}")
                continue
            fl = 2 * M * N * K / (us * 1e-6)
            by = (N * K + M * K + 2 * M * N) / (us * 1e-6)
            print(f"{sn:9s} {M:6d} {impl:7s} {us:9.1f} {fl/1e12:7.1f} {100*fl/PEAK:5.1f} {by/1e9:7.0f} {100*by/BW:5.1f}",
                  flush=True)
        del ws
        torch.cuda.empty_cache()
