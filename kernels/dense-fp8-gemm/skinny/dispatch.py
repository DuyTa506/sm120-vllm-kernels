"""Runtime dispatch used by the benchmarks / correctness test (mirrors the vLLM patch).
dense_fp8_mm(a, b, sa, sb): a [M,K] e4m3, b [K,N] e4m3 column-major (vLLM layout, = w.t()),
sa [M,1] fp32 per-token, sb [N] or [N,1] fp32 per-channel -> bf16 [M,N]."""
import torch

MAX_M = 64
_mod = None


def _skinny():
    global _mod
    if _mod is None:
        from skinny.build import load
        _mod = load()
    return _mod


def pick(M, N, K):
    """(warps splitting K inside a block, 128-wide k-blocks in flight per warp), or None to keep
    vLLM's cutlass_scaled_mm. Tuned on RTX PRO 6000 Blackwell (cold weights, CUDA graphs) for the
    Qwen3.6-35B-A3B dense FP8 shapes; see kernels/dense-fp8-gemm/README.md."""
    if M <= 16:
        if N >= 65536 or K >= 4096:
            return (8, 1)  # lm_head; out_proj / o_proj (N=2048, K=4096)
        return (1, 4)  # in_proj_qkvz (N=12288), qkv_proj (N=9216)
    if M <= 64:
        if N >= 65536:
            return (1, 4) if M <= 48 else None
        if K >= 4096:
            return (8, 1)
        if N >= 12288:
            return (2, 4)
    return None


def supported(a, b, sa, sb):
    M, K = a.shape
    N = b.shape[1]
    return (M <= MAX_M and N % 16 == 0 and K % 1024 == 0 and sa.numel() == M and sb.numel() == N
            and b.stride(0) == 1 and b.stride(1) == K and a.is_contiguous())


def dense_fp8_mm(a, b, sa, sb, out_dtype=torch.bfloat16):
    from vllm import _custom_ops as ops
    if out_dtype == torch.bfloat16 and supported(a, b, sa, sb) and pick(a.shape[0], b.shape[1], a.shape[1]):
        M, K = a.shape
        N = b.shape[1]
        out = torch.empty(M, N, dtype=torch.bfloat16, device=a.device)
        kw, kb = pick(M, N, K)
        _skinny().mm(out, a, b.t(), sa.reshape(-1), sb.reshape(-1), kw, kb)
        return out
    return ops.cutlass_scaled_mm(a, b, sa, sb, out_dtype)
