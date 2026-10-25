"""Single-kernel Q quantisation (fallback if the in-kernel BF16 Q path is not used).

Bitwise equal to vLLM's ``(q / q_scale).clamp(-448, 448).to(float8_e4m3fn)``:
torch evaluates bf16 / cpu-scalar as bf16(x * fp32(1/q_scale)).
"""

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _q_quant_kernel(x_ptr, y_ptr, n, mul, SCALED: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    x = tl.load(x_ptr + offs, mask=m, other=0.0)
    # exact bf16 -> f32 via bit shift (an FTZ'd convert/min/max would flush bf16
    # subnormals); the f32 -> e4m3 convert is cvt.rn.satfinite == clamp(+-448) + RNE.
    xf = (x.to(tl.uint16, bitcast=True).to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    if SCALED:
        xb = (xf * mul).to(tl.bfloat16)
        xf = (xb.to(tl.uint16, bitcast=True).to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    tl.store(y_ptr + offs, xf.to(tl.float8e4nv), mask=m)


def q_quant(q: torch.Tensor, q_scale: float) -> torch.Tensor:
    assert q.dtype == torch.bfloat16 and q.is_contiguous()
    y = torch.empty(q.shape, dtype=torch.float8_e4m3fn, device=q.device)
    n = q.numel()
    BLOCK = 4096
    mul = float(np.float32(1.0) / np.float32(q_scale))
    _q_quant_kernel[(triton.cdiv(n, BLOCK),)](q, y, n, mul, SCALED=q_scale != 1.0,
                                              BLOCK=BLOCK, num_warps=8)
    return y
