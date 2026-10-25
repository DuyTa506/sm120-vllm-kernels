"""Microbench: FP8 lm_head GEMM (vLLM cutlass_scaled_mm, per-token act scale, per-channel weight scale)
at decode M, full vocab vs a vocab prefix (draft-vocab pruning), plus achieved bandwidth.
Uses the real lm_head weight from the checkpoint.
usage (in the vLLM image): python3 kernels/decode-roofline/bench_lmhead.py
"""
import glob, json, torch
from safetensors import safe_open
from vllm import _custom_ops as ops

snap = glob.glob("/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/*/")[0]
idx = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
with safe_open(snap + idx["lm_head.weight"], "pt", device="cuda") as f:
    W = f.get_tensor("lm_head.weight")              # [V, K] fp8
with safe_open(snap + idx["lm_head.weight_scale"], "pt", device="cuda") as f:
    Ws = f.get_tensor("lm_head.weight_scale").float()  # [V, 1]
V, K = W.shape
print("lm_head", tuple(W.shape), W.dtype)
Wt = W.t()  # [K, V] column-major view, what cutlass_scaled_mm expects


def bench(fn, n=50, warm=10):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


# flush L2 between runs is unnecessary: lm_head (509 MB) >> L2 (128 MB) for full vocab;
# for small prefixes we rotate through several copies to avoid L2 residency.
torch.manual_seed(0)
for M in (1, 3, 8, 64, 192):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    xq, xs = ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)
    for N in (V, 131072, 65536, 32768):
        copies = max(1, (512 << 20) // (N * K))
        Ws_ = [Wt[:, :N].clone().t().contiguous().t() if c else Wt[:, :N] for c in range(copies)]
        Ss_ = [Ws[:N].contiguous() for _ in range(copies)]
        it = [0]

        def fn():
            i = it[0] % copies; it[0] += 1
            ops.cutlass_scaled_mm(xq, Ws_[i], xs, Ss_[i], torch.bfloat16)
        us = bench(fn)
        print(f"M={M:4d} N={N:6d}: {us:7.1f} us  {N*K/us/1e3:6.0f} GB/s (weight bytes only)")
    # correctness of slicing: prefix of full logits == sliced GEMM
    full = ops.cutlass_scaled_mm(xq, Wt, xs, Ws, torch.bfloat16)
    part = ops.cutlass_scaled_mm(xq, Wt[:, :32768], xs, Ws[:32768].contiguous(), torch.bfloat16)
    print(f"   slice check M={M}: max|diff| {(full[:, :32768]-part).abs().max().item():.3g}")
