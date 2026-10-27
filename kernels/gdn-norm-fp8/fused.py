"""UNVALIDATED standalone SM120 GDN norm/gate/token-FP8 experiment.

No serving integration. GPU correctness, generated code and timing are pending.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _norm_gate_fp8(
    X, Z, W, Q, S,
    stride_x_token: tl.constexpr,
    stride_z_token: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.arange(0, 32)
    channel = tl.arange(0, 128)
    offset = head[:, None] * 128 + channel[None, :]
    x = tl.load(X + token * stride_x_token + offset).to(tl.float32)
    variance = tl.sum(x * x, axis=1) * (1.0 / 128.0)
    rstd = tl.rsqrt(variance + EPS)
    w = tl.load(W + channel).to(tl.float32)
    z = tl.load(Z + token * stride_z_token + offset).to(tl.float32)
    y = ((x * rstd[:, None]) * w[None, :]) * (z * tl.sigmoid(z))

    # This lossy round-trip is part of the baseline contract even though Y is
    # no longer stored to global memory. Inspect IR before claiming equality.
    y = y.to(tl.bfloat16).to(tl.float32)
    per_head_amax = tl.max(tl.abs(y), axis=1)
    token_amax = tl.max(per_head_amax, axis=0)
    scale = tl.maximum(token_amax * (1.0 / 448.0), 1.0 / (448.0 * 512.0))
    scaled = y * (1.0 / scale)
    q = tl.minimum(tl.maximum(scaled, -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Q + token * 4096 + offset, q)
    tl.store(S + token, scale)


def fused_norm_gate_fp8(x, z, weight, *, out=None, num_warps=8):
    """Return FP8 [T,4096] and FP32 [T,1], with strided token input support.

    Fixed BF16 x/z/weight, 32 heads of width 128, epsilon 1e-6, SiLU,
    norm_before_gate=True, no bias, no scale upper bound, no padding.
    Caller-provided outputs must not alias inputs. No input is mutated.
    """
    if x.ndim != 3 or x.shape[1:] != (32, 128) or z.shape != x.shape:
        raise ValueError("x and z must have shape [T,32,128]")
    if any(t.dtype != torch.bfloat16 for t in (x, z, weight)):
        raise ValueError("x, z and weight must be BF16")
    if not x.is_cuda or z.device != x.device or weight.device != x.device:
        raise ValueError("all inputs must share one CUDA device")
    if weight.shape != (128,) or not weight.is_contiguous():
        raise ValueError("weight must be contiguous [128]")
    if x.stride()[1:] != (128, 1) or z.stride()[1:] != (128, 1):
        raise ValueError("head/channel dimensions must be contiguous; token stride may vary")
    if num_warps not in (4, 8, 16):
        raise ValueError("num_warps must be 4, 8 or 16")
    tokens = x.shape[0]
    if out is None:
        q = torch.empty((tokens, 4096), dtype=torch.float8_e4m3fn, device=x.device)
        scale = torch.empty((tokens, 1), dtype=torch.float32, device=x.device)
    else:
        q, scale = out
        if (q.shape != (tokens, 4096) or q.dtype != torch.float8_e4m3fn
                or scale.shape != (tokens, 1) or scale.dtype != torch.float32
                or q.device != x.device or scale.device != x.device
                or not q.is_contiguous() or not scale.is_contiguous()):
            raise ValueError("out must be contiguous CUDA FP8 [T,4096], FP32 [T,1]")
    if tokens:
        with torch.cuda.device(x.device):
            _norm_gate_fp8[(tokens,)](
                x, z, weight, q, scale, x.stride(0), z.stride(0), 1e-6,
                num_warps=num_warps,
            )
    return q, scale
