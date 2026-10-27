"""UNVALIDATED explicit-layout candidate matching installed FLA row reductions.

Isolated from fused.py and serving. Offline compilation is not GPU validation.
"""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language import BlockedLayout, SliceLayout


@gluon.jit
def _norm_gate_fp8_explicit(
    X, Z, W, Q, S,
    stride_x_token: gl.constexpr, stride_z_token: gl.constexpr,
    EPS: gl.constexpr, VALUES_PER_LANE: gl.constexpr, WARPS: gl.constexpr,
):
    # Installed FLA uses 4 values/lane for ROWS_PER_BLOCK=1 and 8 for 2/4.
    # Keep that per-head reduction topology independent of CTA warp count.
    layout: gl.constexpr = BlockedLayout(
        [1, VALUES_PER_LANE], [VALUES_PER_LANE // 4, 128 // VALUES_PER_LANE],
        [WARPS, 1], [1, 0])
    token = gl.program_id(0)
    head = gl.arange(0, 32, layout=SliceLayout(1, layout))
    channel = gl.arange(0, 128, layout=SliceLayout(0, layout))
    offset = head[:, None] * 128 + channel[None, :]
    x = gl.load(X + token * stride_x_token + offset).to(gl.float32)
    sum_squares = gl.sum(x * x, axis=1)
    # FLA emits div.full.f32 by 128 followed by a separate epsilon addition.
    # Preserve that boundary as well as the reduction topology.
    variance = gl.inline_asm_elementwise(
        "div.full.f32 $0, $1, 0f43000000;", constraints="=f,f",
        args=[sum_squares], dtype=gl.float32, is_pure=True, pack=1)
    rstd = gl.rsqrt(variance + EPS)
    w = gl.load(W + channel).to(gl.float32)
    z = gl.load(Z + token * stride_z_token + offset).to(gl.float32)
    # Identical expression to triton.language.sigmoid used by installed FLA.
    sigmoid = 1.0 / (1.0 + gl.exp(-z))
    y = ((x * rstd[:, None]) * w[None, :]) * (z * sigmoid)
    y = y.to(gl.bfloat16).to(gl.float32)
    token_amax = gl.max(gl.max(gl.abs(y), axis=1), axis=0)
    scale = gl.maximum(token_amax * (1.0 / 448.0), 1.0 / (448.0 * 512.0))
    q = gl.minimum(gl.maximum(y * (1.0 / scale), -448.0), 448.0).to(gl.float8e4nv)
    gl.store(Q + token * 4096 + offset, q)
    gl.store(S + token, scale)


def fused_norm_gate_fp8_explicit(x, z, weight, *, out=None, num_warps=8,
                                 reference_rows_per_block=None):
    """Experimental replacement with the same input/output contract as fused.py.

The row-count specialization follows installed FLA, including short inputs.
Every requested warp count still needs exact GPU validation independently.
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
        raise ValueError("head/channel dimensions must be contiguous")
    if num_warps not in (4, 8, 16):
        raise ValueError("num_warps must be 4, 8 or 16")
    tokens = x.shape[0]
    if reference_rows_per_block is None and tokens:
        from vllm.third_party.flash_linear_attention.ops.layernorm_guard import calc_rows_per_block
        reference_rows_per_block = calc_rows_per_block(tokens * 32, x.device)
    if tokens and reference_rows_per_block not in (1, 2, 4):
        raise ValueError("Unsupported installed FLA rows-per-block specialization")
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
            _norm_gate_fp8_explicit[(tokens,)](
                x, z, weight, q, scale, x.stride(0), z.stride(0), 1e-6,
                4 if reference_rows_per_block == 1 else 8, num_warps, num_warps=num_warps)
    return q, scale
