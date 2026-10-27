"""(vLLM copy of kernels/fp8-attn-glue/glue_fmha.py) Glue-free variant of FlashInfer's SM120 FP8 FMHA (paged, causal) for vLLM.

Two changes versus flashinfer.cute_dsl.attention.fmha.sm120 (FlashInfer 0.7.0.post1):

1. Strided K/V pools.  FlashInfer compiles the paged kernel with compact
   inner strides ``(sym, page*D, D, 1)``.  vLLM packs K|V per row, so its K and
   V views have strides ``(Hkv*page*2D, page*2D, 2D, 1)`` (HND) and the vLLM
   glue had to ``index_select`` every page of the batch into a compact copy for
   each attention layer.  K/V are only ever touched through TMA descriptors that
   are built at launch from the runtime view, so compiling the fake K/V tensors
   with symbolic strides lets the kernel read vLLM's cache in place.  The block
   table then holds absolute vLLM block ids instead of indices into the copy.

2. BF16 Q quantised inside the kernel.  The compute warps already load Q
   straight from global memory into MMA A-fragment registers (4 FP8 values
   per lane per row-quad).  The variant loads 4 BF16 values (8 bytes) instead
   and converts them with ``cvt.rn.satfinite.e4m3x2.f32``.  This reproduces
   vLLM's ``(q / q_scale).clamp(-448, 448).to(float8_e4m3fn)`` bit for bit:
   BF16->F32 is exact, satfinite == clamp to +-448 (incl. inf), RNE on both
   sides.  For q_scale != 1 the torch path computes ``bf16(x * (1/q_scale))``
   (PyTorch divides by a CPU scalar via its reciprocal) and that rounding is
   emulated as well.  q_scale is a compile-time constant (vLLM uses 1.0 for this
   checkpoint, which has no q_scale tensor).

Nothing else in the kernel changes: same tiles, pipeline, softmax and epilogue,
so the output is bitwise identical to the existing path (see correctness.py).
"""

from __future__ import annotations

import functools
import numpy as np

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cute.typing import Int32
from cutlass.experimental import primitives as prims

from flashinfer.cute_dsl.attention.fmha.sm120.fmha_prefill_fp8_tma import (
    SM120FusedMultiHeadAttentionFP8ForwardTMA as _Base,
)

_CT = {
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
}


@cute.jit
def bf16x4_to_e4m3x4(v: cutlass.Int64) -> cutlass.Int32:
    """Four packed BF16 (8 bytes, little endian) -> four packed E4M3 (RNE, satfinite)."""
    return cute.arch.inline_ptx(
        (
            "{\n"
            "  .reg .b32 w0, w1, u0, u1, u2, u3;\n"
            "  .reg .f32 f0, f1, f2, f3;\n"
            "  .reg .b16 lo, hi;\n"
            "  mov.b64 {w0, w1}, $1;\n"
            "  shl.b32 u0, w0, 16;\n"
            "  and.b32 u1, w0, 0xFFFF0000;\n"
            "  shl.b32 u2, w1, 16;\n"
            "  and.b32 u3, w1, 0xFFFF0000;\n"
            "  mov.b32 f0, u0;\n"
            "  mov.b32 f1, u1;\n"
            "  mov.b32 f2, u2;\n"
            "  mov.b32 f3, u3;\n"
            "  cvt.rn.satfinite.e4m3x2.f32 lo, f1, f0;\n"
            "  cvt.rn.satfinite.e4m3x2.f32 hi, f3, f2;\n"
            "  mov.b32 $0, {lo, hi};\n"
            "}"
        ),
        write_only_types=[cutlass.Int32],
        read_only_args=[v],
    )


@cute.jit
def bf16x4_scaled_to_e4m3x4(v: cutlass.Int64, mul: cutlass.Float32) -> cutlass.Int32:
    """Same as above with torch's ``bf16(x * mul)`` rounding step in between."""
    return cute.arch.inline_ptx(
        (
            "{\n"
            "  .reg .b32 w0, w1, u0, u1, u2, u3;\n"
            "  .reg .f32 f0, f1, f2, f3;\n"
            "  .reg .b16 lo, hi;\n"
            "  mov.b64 {w0, w1}, $1;\n"
            "  shl.b32 u0, w0, 16;\n"
            "  and.b32 u1, w0, 0xFFFF0000;\n"
            "  shl.b32 u2, w1, 16;\n"
            "  and.b32 u3, w1, 0xFFFF0000;\n"
            "  mov.b32 f0, u0;\n"
            "  mov.b32 f1, u1;\n"
            "  mov.b32 f2, u2;\n"
            "  mov.b32 f3, u3;\n"
            "  mul.rn.f32 f0, f0, $2;\n"
            "  mul.rn.f32 f1, f1, $2;\n"
            "  mul.rn.f32 f2, f2, $2;\n"
            "  mul.rn.f32 f3, f3, $2;\n"
            "  cvt.rn.bf16x2.f32 w0, f1, f0;\n"
            "  cvt.rn.bf16x2.f32 w1, f3, f2;\n"
            "  shl.b32 u0, w0, 16;\n"
            "  and.b32 u1, w0, 0xFFFF0000;\n"
            "  shl.b32 u2, w1, 16;\n"
            "  and.b32 u3, w1, 0xFFFF0000;\n"
            "  mov.b32 f0, u0;\n"
            "  mov.b32 f1, u1;\n"
            "  mov.b32 f2, u2;\n"
            "  mov.b32 f3, u3;\n"
            "  cvt.rn.satfinite.e4m3x2.f32 lo, f1, f0;\n"
            "  cvt.rn.satfinite.e4m3x2.f32 hi, f3, f2;\n"
            "  mov.b32 $0, {lo, hi};\n"
            "}"
        ),
        write_only_types=[cutlass.Int32],
        read_only_args=[v, mul],
    )


class GlueFMHA(_Base):
    """SM120 FP8 FMHA that optionally takes BF16 Q and strided paged K/V."""

    def __init__(self, *args, q_bf16: bool = False, q_scale: float = 1.0, **kw):
        super().__init__(*args, **kw)
        self.q_bf16 = q_bf16
        # torch: bf16 / python-float  ->  bf16(x * float32(1/float32(q_scale)))
        self.q_mul = float(np.float32(1.0) / np.float32(q_scale))
        self.q_scaled = q_scale != 1.0

    def load_q_tile(self, basic_params):  # trace-time dispatch
        if not self.q_bf16:
            return _Base.load_q_tile(self, basic_params)
        return self._load_q_tile_bf16(basic_params)

    @cute.jit
    def _load_q_tile_bf16(self, basic_params):
        def load_q_fragment(col0_in_cta: cutlass.Int32):
            q_regs_per_frag = [
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
            ]
            for row_quad in cutlass.range_constexpr(self.MMA_TILER[0] // 4):
                row_in_cta = (
                    basic_params.q_warp_row0 + row_quad * 4 + basic_params.lane_div8
                )
                col_in_cta = col0_in_cta + basic_params.lane_mod8 * 4
                cur_q_seq_idx = basic_params.q_seq_idx + row_in_cta
                if (
                    cur_q_seq_idx < basic_params.seqlen_q
                    and col_in_cta < basic_params.head_dim
                ):
                    q_src = (
                        basic_params.q_ptr
                        + basic_params.q_head_base
                        + cur_q_seq_idx * basic_params.q_seq_stride
                        + col_in_cta
                    )
                    q_raw = prims.load_ext(
                        q_src,
                        dtype=cutlass.Int64,
                        l2_cache_hint=basic_params.q_l2_cache_hint,
                    )
                    if cutlass.const_expr(self.q_scaled):
                        q_regs_per_frag[row_quad] = bf16x4_scaled_to_e4m3x4(
                            q_raw, cutlass.Float32(self.q_mul)
                        )
                    else:
                        q_regs_per_frag[row_quad] = bf16x4_to_e4m3x4(q_raw)
            return q_regs_per_frag

        q_regs = cutlass.Array(cutlass.Int32, self.qk_d_frags * 4, alignment=16)
        src_lane_lo = (basic_params.lane_div4 % 4) * 8 + basic_params.lane_mod4
        src_lane_hi = (basic_params.lane_div4 % 4) * 8 + (basic_params.lane_mod4 + 4)

        q_regs_offset = 0
        col0_in_cta = 0
        for _ in cutlass.range_constexpr(self.qk_d_frags):
            q_rows = load_q_fragment(col0_in_cta)
            q_rows_lo = [
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=q_rows[i],
                    offset=src_lane_lo,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                )
                for i in range(4)
            ]
            q_rows_hi = [
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=q_rows[i],
                    offset=src_lane_hi,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.IDX,
                )
                for i in range(4)
            ]
            if basic_params.lane_div4 < 4:
                q_regs[q_regs_offset + 0] = q_rows_lo[0]
                q_regs[q_regs_offset + 1] = q_rows_lo[2]
                q_regs[q_regs_offset + 2] = q_rows_hi[0]
                q_regs[q_regs_offset + 3] = q_rows_hi[2]
            else:
                q_regs[q_regs_offset + 0] = q_rows_lo[1]
                q_regs[q_regs_offset + 1] = q_rows_lo[3]
                q_regs[q_regs_offset + 2] = q_rows_hi[1]
                q_regs[q_regs_offset + 3] = q_rows_hi[3]
            col0_in_cta += self.MMA_TILER[2]
            q_regs_offset += 4
        return q_regs


def _compile(
    q_dtype: torch.dtype,
    out_dtype: torch.dtype,
    hq: int,
    hkv: int,
    d: int,
    page: int,
    kv_tile: int = 128,
    q_tile: int = 128,
    balanced: bool = True,
    strided_kv: bool = True,
    q_scale: float = 1.0,
):
    q_bf16 = q_dtype == torch.bfloat16
    if not q_bf16 and q_dtype != torch.float8_e4m3fn:
        raise ValueError(q_dtype)
    fmha = GlueFMHA(
        in_dtype=cutlass.Float8E4M3FN,
        out_dtype=_CT[out_dtype],
        is_causal=True,
        head_tile=d,
        kv_tile=kv_tile,
        q_tile=q_tile,
        use_paged_kv=True,
        num_tokens_per_page=page,
        balanced_scheduler=balanced,
        q_bf16=q_bf16,
        q_scale=q_scale,
    )
    sym_tq = cute.sym_int()
    fake_q = make_fake_compact_tensor(
        _CT[q_dtype], (sym_tq, hq, d), stride_order=(2, 1, 0), assumed_align=16
    )
    fake_o = make_fake_compact_tensor(
        _CT[out_dtype], (sym_tq, hq, d), stride_order=(2, 1, 0), assumed_align=16
    )
    sym_np = cute.sym_int()
    if strided_kv:
        # Any 16B-aligned outer strides (vLLM HND: Hkv*page*2D, page*2D, 2D; NHD after
        # transpose: page*Hkv*2D, 2D, Hkv*2D).  Inner dim must be contiguous for TMA.
        kv_stride = (
            cute.sym_int(divisibility=16),
            cute.sym_int(divisibility=16),
            cute.sym_int(divisibility=16),
            1,
        )
    else:
        kv_stride = (cute.sym_int(), page * d, d, 1)
    fake_k = cute.runtime.make_fake_tensor(
        cutlass.Float8E4M3FN, (sym_np, hkv, page, d), stride=kv_stride, assumed_align=16
    )
    fake_v = cute.runtime.make_fake_tensor(
        cutlass.Float8E4M3FN, (sym_np, hkv, page, d), stride=kv_stride, assumed_align=16
    )
    fake_seqlens = make_fake_compact_tensor(Int32, (cute.sym_int(),), assumed_align=4)
    fake_cu_q = make_fake_compact_tensor(Int32, (cute.sym_int(),), assumed_align=4)
    fake_bt = make_fake_compact_tensor(
        Int32, (cute.sym_int(), cute.sym_int()), stride_order=(1, 0), assumed_align=4
    )
    return cute.compile(
        fmha,
        fake_q,
        fake_k,
        fake_v,
        fake_o,
        None,
        cutlass.Float32(1.0),
        cutlass.Float32(1.0),
        make_fake_stream(use_tvm_ffi_env_stream=True),
        fake_seqlens,
        fake_cu_q,
        fake_bt,
        None,
        cutlass.Int32(1),
        True,
        options="--enable-tvm-ffi",
    )


@functools.cache
def get_kernel(q_dtype, out_dtype, hq, hkv, d, page, kv_tile=128, q_tile=128,
               balanced=True, strided_kv=True, q_scale=1.0):
    """Compiled launcher: fn(q, k, v, o, None, scale_log2, v_scale, seqlens_kv,
    cu_seqlens_q, block_tables, None, max_q, use_pdl).

    Uses FlashInfer's CuTe-DSL disk cache (~/.cache/flashinfer) so a server restart
    does not recompile (~8 s per variant)."""
    compile_fn = lambda: _compile(q_dtype, out_dtype, hq, hkv, d, page, kv_tile,  # noqa: E731
                                  q_tile, balanced, strided_kv, q_scale)
    try:
        import os

        from flashinfer.cute_dsl.attention.fmha.sm120 import fmha_prefill_fp8_tma as _base_mod
        from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    except ImportError:
        return compile_fn()
    name = (f"glue_q{str(q_dtype).removeprefix('torch.')}_o{str(out_dtype).removeprefix('torch.')}"
            f"_hq{hq}_hkv{hkv}_d{d}_page{page}_kt{kv_tile}_qt{q_tile}_bal{int(balanced)}"
            f"_strided{int(strided_kv)}_qs{float(q_scale).hex()}").replace(".", "_").replace("+", "p").replace("-", "m")
    return build_and_load_cute_dsl_kernel(
        "sm120_fmha_fp8_glue", name, compile_fn,
        extra_key_files=(os.path.abspath(__file__), os.path.abspath(_base_mod.__file__)),
    )


def absolute_block_table(paged_kv_indices, paged_kv_indptr, max_pages):
    """[B, max_pages] int32 of absolute vLLM block ids (replaces the per-layer gather).

    Entries past a request's last page are never read by the kernel (it clamps the
    logical page to ceil(seqlen/page)-1), they are only clamped to stay in range.
    """
    n = paged_kv_indices.shape[0]
    pos = paged_kv_indptr[:-1, None] + torch.arange(
        max_pages, device=paged_kv_indices.device, dtype=paged_kv_indptr.dtype
    )[None, :]
    pos = pos.clamp_(max=max(n - 1, 0))
    return paged_kv_indices[pos.long()].to(torch.int32)
