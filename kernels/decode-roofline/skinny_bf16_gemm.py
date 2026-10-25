# SPDX-License-Identifier: Apache-2.0
"""Skinny BF16 GEMM for decode-sized batches (M <= 16): y = x @ W^T (+ b).

Why: on SM120, cuBLAS picks `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2` for the small
unquantized BF16 linears of Qwen3.5/3.6 at M=3 (MoE router gate 256x2048, GDN in_proj_ba 64x2048, MTP
BF16 projections). That kernel launches only 4-16 single-warp CTAs and takes ~11.5 us to read
0.25-1 MB (~25-90 GB/s), 70 times per verify forward. This kernel spreads the weight rows over
>=128 programs and streams each row slice once, keeping all M rows of x in registers per k-step.

Numerics: fp32 accumulation, one bf16 rounding of the result (same contract as cuBLAS); only the
summation order differs.
"""
import torch
import triton
import triton.language as tl

MAX_M = 16


@triton.jit
def _skinny_bf16_gemm_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    M, N, K,
    stride_xm, stride_wn, stride_ym,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    pid = tl.program_id(0)
    m0 = tl.program_id(1) * BM
    rn = pid * BN + tl.arange(0, BN)
    rm = m0 + tl.arange(0, BM)
    x_ptr += m0 * stride_xm
    n_mask = rn < N
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        k_mask = rk < K
        w = tl.load(w_ptr + rn[:, None] * stride_wn + rk[None, :],
                    mask=n_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)
        for m in tl.static_range(BM):
            xm = tl.load(x_ptr + m * stride_xm + rk, mask=k_mask & (m0 + m < M), other=0.0).to(tl.float32)
            part = tl.sum(w * xm[None, :], axis=1)  # [BN]
            acc = tl.where(rm[:, None] == m0 + m, acc + part[None, :], acc)
    if HAS_BIAS:
        acc += tl.load(b_ptr + rn, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    tl.store(y_ptr + rm[:, None] * stride_ym + rn[None, :], acc.to(y_ptr.dtype.element_ty),
             mask=(rm[:, None] < M) & n_mask[None, :])


def _config(M: int, N: int, K: int):
    # rows are split into chunks of <= 4 (one grid row each): the weight tile is re-read from L2
    bm = 1
    while bm < min(M, 4):
        bm *= 2
    # aim for >= ~128 programs; keep the [BN, BK] weight tile at ~4K elements
    bn = 1
    while bn < 16 and N // (bn * 2) >= 128:
        bn *= 2
    bk = max(256, min(2048, 4096 // bn))
    bk = min(bk, triton.next_power_of_2(K))
    return bm, bn, bk


def supported(x: torch.Tensor, weight: torch.Tensor, bias) -> bool:
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or not x.is_cuda:
        return False
    if weight.dim() != 2 or weight.stride(1) != 1 or x.shape[-1] != weight.shape[1]:
        return False
    if bias is not None and (bias.dtype != torch.bfloat16 or not bias.is_contiguous()):
        return False
    return True


def skinny_bf16_gemm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """x: [..., K] bf16 with prod(leading) <= MAX_M; weight: [N, K] bf16 (row stride arbitrary)."""
    lead = x.shape[:-1]
    K = x.shape[-1]
    x2 = x.reshape(-1, K)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    M = x2.shape[0]
    N = weight.shape[0]
    y = torch.empty((M, N), dtype=x.dtype, device=x.device)
    if M == 0:
        return y.reshape(*lead, N)
    bm, bn, bk = _config(M, N, K)
    grid = (triton.cdiv(N, bn), triton.cdiv(M, bm))
    _skinny_bf16_gemm_kernel[grid](
        x2, weight, bias if bias is not None else weight, y,
        M, N, K, x2.stride(0), weight.stride(0), y.stride(0),
        HAS_BIAS=bias is not None, BM=bm, BN=bn, BK=bk, num_warps=4,
    )
    return y.reshape(*lead, N)
