# SPDX-License-Identifier: Apache-2.0
"""Skinny-M FP8 GEMM for SM120 decode (gated by VLLM_SM120_SKINNY_FP8=1).

For M <= VLLM_SM120_SKINNY_FP8_MAX_M (default 64) tokens, per-token activation scales and per-channel
weight scales, `torch.ops.vllm.sm120_fp8_scaled_mm` runs a swap-AB tensor-core GEMV that streams the
weight straight from DRAM into mma fragments (see sm120_skinny_fp8.cu); larger M (and anything it
cannot handle) falls through to vLLM's `cutlass_scaled_mm`. The decision is taken inside the op at
run time, so it is invisible to torch.compile and frozen per size by CUDA graph capture.

Epilogue math is identical to the CUTLASS ScaledEpilogue (bf16_rn(sa * (sb * acc)), fp32 acc); only
the fp32 summation order of the K reduction differs (bf16-rounding-level differences, deterministic).
The extension is JIT-compiled once with nvcc into $VLLM_CACHE_ROOT/sm120_skinny_fp8.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_SM120_SKINNY_FP8", "0") == "1"
MAX_M = int(os.environ.get("VLLM_SM120_SKINNY_FP8_MAX_M", "64"))

_mod = None


def _load():
    global _mod
    if _mod is None:
        from torch.utils.cpp_extension import load

        root = os.environ.get("VLLM_CACHE_ROOT", os.path.expanduser("~/.cache/vllm"))
        bd = os.path.join(root, "sm120_skinny_fp8")
        os.makedirs(bd, exist_ok=True)
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sm120_skinny_fp8.cu")
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
        _mod = load(
            name="sm120_skinny_fp8",
            sources=[src],
            build_directory=bd,
            extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_120a,code=sm_120a"],
            verbose=False,
        )
        logger.info("SM120 skinny FP8 GEMM loaded (M <= %d)", MAX_M)
    return _mod


def _pick(M, N, K):
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


def _sm120_fp8_scaled_mm(a: torch.Tensor, b: torch.Tensor, sa: torch.Tensor, sb: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    N = b.shape[1]
    cfg = _pick(M, N, K) if M <= MAX_M else None
    if cfg is not None and sa.numel() == M and a.is_contiguous() and a.data_ptr() % 16 == 0:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
        _load().mm(out, a, b.t(), sa.reshape(-1), sb.reshape(-1), cfg[0], cfg[1])
        return out
    out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    torch.ops._C.cutlass_scaled_mm(out, a, b, sa, sb, None)
    return out


def _sm120_fp8_scaled_mm_fake(a, b, sa, sb):
    return torch.empty((a.shape[0], b.shape[1]), dtype=torch.bfloat16, device=a.device)


if ENABLED:
    direct_register_custom_op(
        op_name="sm120_fp8_scaled_mm",
        op_func=_sm120_fp8_scaled_mm,
        fake_impl=_sm120_fp8_scaled_mm_fake,
    )
    _load()  # compile/load eagerly at model init, not inside CUDA graph capture
