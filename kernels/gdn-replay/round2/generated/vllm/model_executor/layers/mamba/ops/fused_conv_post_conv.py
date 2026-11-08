# SPDX-License-Identifier: Apache-2.0
"""Fused GDN prefill: causal conv1d (+SiLU, +conv_state update) and post-conv prep.

Replaces, for the Qwen3.5/3.6 GDN prefill path,

    out = causal_conv1d_fn(mixed_qkv.T, w, bias, conv_states, query_start_loc,
                           cache_indices, has_initial_state, metadata=...).T
    q, k, v, g, beta = fused_post_conv_prep(out, a, b, A_log, dt_bias, ...)

with one Triton kernel that never writes the conv output to HBM.

Work decomposition is identical to ``_causal_conv1d_fwd_kernel``: axis "chunk" walks
the (sequence, 8-token chunk) list built by ``compute_causal_conv1d_metadata`` (the
``batch_ptr`` / ``token_chunk_offset_ptr`` buffers already in the GDN metadata), so
a program never straddles two sequences and the conv_state read/write, the
prefix-caching (APC) intermediate-state stores and the null-block skip are the
same code as the existing kernel. The other axis is a 256-channel group:

  * groups [0, H):        q head h and k head h (2 x K channels) -> conv, SiLU,
                          per-row L2 norm of each half, store into q / k
  * groups [H, H+HV/2):   two v heads (2 x V channels) -> conv, SiLU, store into v,
                          plus g (or exp(g)) and beta for those two heads

Numerics follow the two-kernel path bit for bit: the conv accumulates in fp32 with
bf16 products in the same order, the SiLU output is rounded to the activation dtype
(that is what the old path stored to HBM) before the fp32 L2 norm.

Restrictions (checked by the wrapper): kernel width 4, K == V, K a power of two,
activation silu, x row-major in channels (stride(-1) == 1).
"""

from __future__ import annotations

import numpy as np
import torch

try:  # inside vLLM
    from vllm.triton_utils import tl, triton
except Exception:  # pragma: no cover - standalone use
    import triton
    import triton.language as tl

PAD_SLOT_ID = -1
NULL_BLOCK_ID = 0
BLOCK_M = 8  # must match compute_causal_conv1d_metadata


@triton.jit(do_not_specialize_on_alignment=["num_cache_lines"])
def _fused_conv_post_conv_kernel(
    x_ptr,  # [T, >=dim] token-major, channel stride 1
    w_ptr,  # [dim, width]
    bias_ptr,
    conv_states_ptr,  # [num_cache_lines, dim, state_len] (any strides)
    cache_indices_ptr,
    has_initial_states_ptr,
    query_start_loc_ptr,
    batch_ptr,
    token_chunk_offset_ptr,
    block_idx_first_scheduled_token,
    block_idx_last_scheduled_token,
    initial_state_idx,
    num_computed_tokens,
    a_ptr,  # [T, HV]
    b_ptr,  # [T, HV]
    A_log_ptr,  # [HV]
    dt_bias_ptr,  # [HV]
    qk_ptr,  # [2, Tp, H, K] (q = qk[0], k = qk[1])
    v_ptr,  # [Tp, HV, V]
    g_ptr,  # [Tp, HV] fp32
    beta_ptr,  # [Tp, HV] fp32
    raw_ptr,  # [num_raw_tokens, dim] conv output for the leading "raw" sequences
    num_cache_lines,
    num_raw_seqs,
    num_raw_tokens,
    num_prefill_tokens,  # Tp
    stride_x_token: tl.int64,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_cs_seq: tl.constexpr,
    stride_cs_dim: tl.constexpr,
    stride_cs_tok: tl.constexpr,
    stride_cache_indices: tl.constexpr,
    stride_a_tok,
    stride_b_tok,
    stride_raw_tok: tl.int64,
    stride_block_m: tl.constexpr,
    pad_slot_id: tl.constexpr,
    null_block_id: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    HAS_NULL_BLOCK: tl.constexpr,
    HAS_RAW: tl.constexpr,
    APPLY_L2NORM: tl.constexpr,
    L2NORM_EPS: tl.constexpr,
    OUTPUT_G_EXP: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    SUB: tl.constexpr,  # 8-token chunks per program (programs with chunk % SUB != 0 exit)
    GROUP_FASTEST: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    state_len: tl.constexpr = 3  # KERNEL_WIDTH - 1
    NP2_STATELEN: tl.constexpr = 4
    HK: tl.constexpr = H * K
    dim: tl.constexpr = 2 * HK + HV * K

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    if GROUP_FASTEST:
        pid_chunk = tl.program_id(1)
        pid_grp = tl.program_id(0)
    else:
        pid_chunk = tl.program_id(0)
        pid_grp = tl.program_id(1)

    idx_seq = tl.load(batch_ptr + pid_chunk).to(tl.int64)
    chunk_offset = tl.load(token_chunk_offset_ptr + pid_chunk)
    if (idx_seq == pad_slot_id) | (chunk_offset % SUB != 0):
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
        return

    # ---- channel layout of this program: [2 halves, K] ----
    is_qk = pid_grp < H
    if is_qk:
        ch_base = pid_grp * K
        ch_half = HK
    else:
        ch_base = 2 * HK + (pid_grp - H) * (2 * K)
        ch_half = K
    offs_h = tl.arange(0, 2)
    offs_k = tl.arange(0, K)
    feats = ch_base + offs_h[:, None] * ch_half + offs_k[None, :]  # [2, K]
    feats3 = feats[None, :, :]  # [1, 2, K]

    sequence_start_index = tl.load(query_start_loc_ptr + idx_seq)
    sequence_end_index = tl.load(query_start_loc_ptr + idx_seq + 1)
    seqlen = sequence_end_index - sequence_start_index

    B_size: tl.constexpr = stride_block_m * BLOCK_M

    if IS_APC_ENABLED:
        current_first_index = tl.load(block_idx_first_scheduled_token + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
        sequence_completed_index = tl.load(num_computed_tokens + idx_seq)
        sequence_completed_offset_token = sequence_completed_index % B_size
        seq_completed_offset = B_size - sequence_completed_offset_token
        seq_end_offset = (seqlen - seq_completed_offset) % B_size
        last_full_block_token_index = sequence_end_index - seq_end_offset
        if seq_end_offset == 0:
            last_full_block_token_index = last_full_block_token_index - B_size
        n_block_to_fill = current_last_index - current_first_index
        conv_state_init_index = tl.load(initial_state_idx + idx_seq)
    else:
        n_block_to_fill = 0
        current_last_index = 0
        conv_state_init_index = 0
        current_first_index = 0
        last_full_block_token_index = 0

    token_offset = BLOCK_M * chunk_offset

    conv_states_input_coord = tl.load(
        cache_indices_ptr + idx_seq * stride_cache_indices + conv_state_init_index
    ).to(tl.int64)
    if HAS_NULL_BLOCK:  # noqa
        if conv_states_input_coord == null_block_id:
            if launch_pdl:
                tl.extra.cuda.gdc_launch_dependents()
            return

    cs_feat = feats3 * stride_cs_dim  # [1, 2, K]
    # Main tile is 2-D [R, K]: row r = half (r // TILE_M) x token (r % TILE_M), so the
    # per-head L2-norm reduction runs over a [rows, K] tile like the reference kernel.
    TILE_M: tl.constexpr = BLOCK_M * SUB
    R: tl.constexpr = 2 * TILE_M
    offs_r = tl.arange(0, R)
    half_r = offs_r // TILE_M
    t_loc = token_offset + offs_r % TILE_M  # [R] position inside the sequence
    row_ok = t_loc < seqlen
    feats2 = (ch_base + half_r * ch_half)[:, None] + offs_k[None, :]  # [R, K]
    x_seq = x_ptr + sequence_start_index * stride_x_token

    # ---- 4 shifted input tiles: xs_j[t] = x[t - j] (j = 0..3) ----
    t3 = t_loc[:, None]
    m3 = row_ok[:, None]
    x0 = tl.load(x_seq + t3 * stride_x_token + feats2, mask=m3, other=0.0)
    x1 = tl.load(x_seq + (t3 - 1) * stride_x_token + feats2, mask=m3 & (t3 >= 1), other=0.0)
    x2 = tl.load(x_seq + (t3 - 2) * stride_x_token + feats2, mask=m3 & (t3 >= 2), other=0.0)
    x3 = tl.load(x_seq + (t3 - 3) * stride_x_token + feats2, mask=m3 & (t3 >= 3), other=0.0)

    if chunk_offset == 0:
        load_init_state = tl.load(has_initial_states_ptr + idx_seq).to(tl.int1)
        cs_in = conv_states_ptr + conv_states_input_coord * stride_cs_seq + cs_feat
        cs_in2 = conv_states_ptr + conv_states_input_coord * stride_cs_seq + feats2 * stride_cs_dim
        if load_init_state:
            # rows t < j take conv_state[:, state_len + t - j]
            s1 = tl.load(cs_in2 + (state_len + t3 - 1) * stride_cs_tok, mask=m3 & (t3 < 1), other=0.0)
            s2 = tl.load(cs_in2 + (state_len + t3 - 2) * stride_cs_tok, mask=m3 & (t3 < 2), other=0.0)
            s3 = tl.load(cs_in2 + (state_len + t3 - 3) * stride_cs_tok, mask=m3 & (t3 < 3), other=0.0)
            x1 = tl.where(t3 < 1, s1, x1)
            x2 = tl.where(t3 < 2, s2, x2)
            x3 = tl.where(t3 < 3, s3, x3)

        # ---- conv_state update (same cases as _causal_conv1d_fwd_kernel) ----
        idx_tokens_conv = tl.arange(0, NP2_STATELEN)
        tc3 = idx_tokens_conv[:, None, None]
        smask = tc3 < state_len
        if state_len <= seqlen:
            idx_tokens_last = (seqlen - state_len) + tc3
            loaded_x = tl.load(
                x_seq + idx_tokens_last * stride_x_token + feats3,
                mask=(idx_tokens_last >= 0) & (idx_tokens_last < seqlen),
                other=0.0,
            )
            conv_states_output_coord = tl.load(
                cache_indices_ptr + idx_seq * stride_cache_indices + current_last_index
            ).to(tl.int64)
            tl.debug_barrier()
            tl.store(
                conv_states_ptr
                + conv_states_output_coord * stride_cs_seq
                + cs_feat
                + tc3 * stride_cs_tok,
                loaded_x,
                mask=smask,
            )
        else:
            VAL = state_len - seqlen
            loaded_x = tl.load(
                x_seq + (tc3 - VAL) * stride_x_token + feats3,
                mask=(tc3 - VAL >= 0) & (tc3 - VAL < seqlen),
                other=0.0,
            )
            if load_init_state:
                old_mask = (conv_states_input_coord < num_cache_lines) & (
                    (tc3 + seqlen) < state_len
                )
                old = tl.load(cs_in + (tc3 + seqlen) * stride_cs_tok, mask=old_mask, other=0.0)
                tl.debug_barrier()
                new_state = tl.where(old_mask, old, loaded_x)
            else:
                new_state = loaded_x
            tl.debug_barrier()
            tl.store(cs_in + tc3 * stride_cs_tok, new_state, mask=smask)
    if IS_APC_ENABLED:
        # chunk c (> 0) of the reference kernel stores the state ending at a
        # B_size-aligned position when c - 1 < n_block_to_fill
        for i in tl.static_range(SUB):
            c = chunk_offset + i
            if (c > 0) & (c * BLOCK_M < seqlen) & ((c - 1) < n_block_to_fill):
                idx_tokens_conv = tl.arange(0, NP2_STATELEN)
                tc3 = idx_tokens_conv[:, None, None]
                idx_tokens_last = (
                    last_full_block_token_index
                    - (n_block_to_fill - c) * B_size
                    - state_len
                ) + tc3
                loaded_x = tl.load(
                    x_ptr + idx_tokens_last * stride_x_token + feats3,
                    mask=idx_tokens_last >= 0,
                    other=0.0,
                )
                conv_states_output_coord = tl.load(
                    cache_indices_ptr
                    + idx_seq * stride_cache_indices
                    + current_first_index
                    + (c - 1)
                ).to(tl.int64)
                tl.debug_barrier()
                tl.store(
                    conv_states_ptr
                    + conv_states_output_coord * stride_cs_seq
                    + cs_feat
                    + tc3 * stride_cs_tok,
                    loaded_x,
                    mask=tc3 < state_len,
                )

    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    # ---- conv (fp32 accumulate, bf16 products, same order as the reference) ----
    wb = w_ptr + feats2 * stride_w_dim
    w0 = tl.load(wb + 0 * stride_w_width)
    w1 = tl.load(wb + 1 * stride_w_width)
    w2 = tl.load(wb + 2 * stride_w_width)
    w3 = tl.load(wb + 3 * stride_w_width)
    if HAS_BIAS:
        acc = tl.load(bias_ptr + feats2).to(tl.float32)
    else:
        acc = tl.zeros((R, K), dtype=tl.float32)
    acc += x3 * w0
    acc += x2 * w1
    acc += x1 * w2
    acc += x0 * w3
    acc = acc / (1 + tl.exp(-acc))  # SiLU
    y = acc.to(x_ptr.dtype.element_ty)  # what the old path stored to HBM

    t_glob = sequence_start_index + t_loc  # [R]
    if HAS_RAW:
        if idx_seq < num_raw_seqs:
            tl.store(raw_ptr + t_glob[:, None] * stride_raw_tok + feats2, y, mask=m3)
            return
        t_out = t_glob - num_raw_tokens
    else:
        t_out = t_glob
    t_out3 = t_out[:, None].to(tl.int64)

    if is_qk:
        yf = y.to(tl.float32)
        if APPLY_L2NORM:
            ss = tl.sum(yf * yf, axis=1)  # [R]
            inv = 1.0 / tl.sqrt(ss + L2NORM_EPS)
            yf = yf * inv[:, None]
        out_off = (
            half_r[:, None].to(tl.int64) * (num_prefill_tokens * HK)
            + t_out3 * HK
            + pid_grp * K
            + offs_k[None, :]
        )
        tl.store(qk_ptr + out_off, yf.to(qk_ptr.dtype.element_ty), mask=m3)
    else:
        i_hv0 = (pid_grp - H) * 2
        hv = i_hv0 + half_r  # [R] v head of each row
        out_off = t_out3 * (HV * K) + hv[:, None] * K + offs_k[None, :]
        tl.store(v_ptr + out_off, y, mask=m3)

        # gating: one scalar per (token, v head) = per row
        A_log_val = tl.load(A_log_ptr + hv).to(tl.float32)
        dt_bias_val = tl.load(dt_bias_ptr + hv).to(tl.float32)
        a_vals = tl.load(a_ptr + t_glob * stride_a_tok + hv, mask=row_ok, other=0).to(tl.float32)
        b_vals = tl.load(b_ptr + t_glob * stride_b_tok + hv, mask=row_ok, other=0).to(tl.float32)
        xg = a_vals + dt_bias_val
        sp = tl.where(xg > 0, xg + tl.log(1.0 + tl.exp(-xg)), tl.log(1.0 + tl.exp(xg)))
        sp = tl.where(xg <= SOFTPLUS_THRESHOLD, sp, xg)
        g_vals = -tl.exp(A_log_val) * sp
        if OUTPUT_G_EXP:
            g_vals = tl.exp(g_vals)
        beta_vals = tl.sigmoid(b_vals)
        gb = t_out * HV + hv
        tl.store(g_ptr + gb, g_vals, mask=row_ok)
        tl.store(beta_ptr + gb, beta_vals, mask=row_ok)


def _program_list(query_start_loc: torch.Tensor, device):
    seqlens = query_start_loc.diff().to("cpu").numpy()
    nums = -(-seqlens // BLOCK_M)
    mlist = np.repeat(np.arange(len(nums)), nums)
    offs = np.concatenate([np.arange(n) for n in nums]) if len(nums) else np.zeros(0)
    batch_ptr = torch.from_numpy(mlist.astype(np.int32)).to(device)
    off_ptr = torch.from_numpy(offs.astype(np.int32)).to(device)
    return int(nums.sum()), batch_ptr, off_ptr


def fused_conv1d_post_conv_prep(
    x: torch.Tensor,  # [T, dim] (rows may be strided, channels contiguous)
    weight: torch.Tensor,  # [dim, width]
    bias: torch.Tensor | None,
    conv_states: torch.Tensor,  # [num_cache_lines, dim, state_len] view
    query_start_loc: torch.Tensor,  # [batch + 1] int32
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    a: torch.Tensor,  # [T, HV]
    b: torch.Tensor,  # [T, HV]
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    apply_l2norm: bool = True,
    output_g_exp: bool = False,
    activation: str | None = "silu",
    metadata=None,
    num_raw_seqs: int = 0,
    num_raw_tokens: int = 0,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    num_computed_tokens: torch.Tensor | None = None,
    block_size_to_align: int = 0,
    pad_slot_id: int = PAD_SLOT_ID,
    null_block_id: int | None = NULL_BLOCK_ID,
    group_fastest: bool = True,
    sub: int = 2,
    launch_pdl: bool | None = None,
):
    """Returns (q, k, v, g, beta, raw).

    q, k: [Tp, H, K]; v: [Tp, HV, V]; g, beta: [Tp, HV] fp32, Tp = T - num_raw_tokens.
    raw: [num_raw_tokens, dim] SiLU(conv) output of the first ``num_raw_seqs``
    sequences (decode rows in a mixed batch, consumed un-normalized), or None.
    conv_states is updated in place exactly like causal_conv1d_fn.
    """
    if isinstance(activation, bool) and activation:
        activation = "silu"
    assert activation in ("silu", "swish")
    T, dim = x.shape
    H, K, V = num_k_heads, head_k_dim, head_v_dim
    HV = A_log.shape[0]
    assert K == V and (K & (K - 1)) == 0
    assert dim == 2 * H * K + HV * V and HV % 2 == 0
    assert weight.shape == (dim, 4), "kernel width 4 only"
    assert x.stride(1) == 1
    assert conv_states.dtype == x.dtype
    Tp = T - num_raw_tokens
    dev = x.device
    qk = torch.empty(2, Tp, H, K, dtype=x.dtype, device=dev)
    v = torch.empty(Tp, HV, V, dtype=x.dtype, device=dev)
    g = torch.empty(Tp, HV, dtype=torch.float32, device=dev)
    beta = torch.empty(Tp, HV, dtype=torch.float32, device=dev)
    raw = (
        torch.empty(num_raw_tokens, dim, dtype=x.dtype, device=dev)
        if num_raw_seqs > 0
        else None
    )
    if T == 0:
        return qk[0], qk[1], v, g, beta, raw

    if metadata is not None:
        tot = metadata.nums_dict[BLOCK_M]["tot"]
        batch_ptr = metadata.batch_ptr
        off_ptr = metadata.token_chunk_offset_ptr
    else:
        tot, batch_ptr, off_ptr = _program_list(query_start_loc, dev)
    if tot == 0:
        return qk[0], qk[1], v, g, beta, raw

    if block_size_to_align is None or block_size_to_align <= 0:
        block_size_to_align = BLOCK_M
    assert block_size_to_align % BLOCK_M == 0

    if launch_pdl is None:
        try:
            from vllm.platforms import current_platform

            launch_pdl = current_platform.is_arch_support_pdl()
        except Exception:
            launch_pdl = False

    n_grp = H + HV // 2
    if tot > 65535:
        group_fastest = False
    grid = (n_grp, tot) if group_fastest else (tot, n_grp)
    _fused_conv_post_conv_kernel[grid](
        x,
        weight,
        bias,
        conv_states,
        cache_indices,
        has_initial_state,
        query_start_loc,
        batch_ptr,
        off_ptr,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx,
        num_computed_tokens,
        a,
        b,
        A_log,
        dt_bias,
        qk,
        v,
        g,
        beta,
        raw if raw is not None else qk,
        conv_states.size(0),
        num_raw_seqs,
        num_raw_tokens,
        Tp,
        x.stride(0),
        weight.stride(0),
        weight.stride(1),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        cache_indices.stride(0),
        a.stride(0),
        b.stride(0),
        raw.stride(0) if raw is not None else 0,
        block_size_to_align // BLOCK_M,
        pad_slot_id,
        null_block_id if null_block_id is not None else 0,
        H=H,
        HV=HV,
        K=K,
        HAS_BIAS=bias is not None,
        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        HAS_NULL_BLOCK=null_block_id is not None,
        HAS_RAW=num_raw_seqs > 0,
        APPLY_L2NORM=apply_l2norm,
        L2NORM_EPS=1e-6,
        OUTPUT_G_EXP=output_g_exp,
        SOFTPLUS_THRESHOLD=20.0,
        BLOCK_M=BLOCK_M,
        SUB=sub,
        GROUP_FASTEST=group_fastest,
        launch_pdl=launch_pdl,
        num_warps=4,
        num_stages=2,
    )
    return qk[0], qk[1], v, g, beta, raw
