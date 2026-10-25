"""Shared helpers: vLLM-like paged FP8 KV pool, metadata, baseline and new glue paths."""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HQ, HKV, D, PAGE = 16, 2, 256, 64
LOG2E = 1.4426950408889634


def make_pool(num_blocks, layout="HND", seed=0, k_amp=3.0, v_amp=2.0, device="cuda"):
    """vLLM FP8 KV cache for one layer and its (K, V) views as the backend sees them.

    vLLM's FlashInfer backend permutes the cache to HND and packs K|V in the last dim:
    kv_cache_permute = [blocks, Hkv, page, 2D]; K, V = kv_cache_permute.split(D, -1).
    For NHD the storage is [blocks, page, Hkv, 2D] and the glue transposes dims 1,2.
    Values: post-RMSNorm/RoPE-like K (heavy-ish tails) and V, all finite.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    if layout == "HND":
        raw = torch.empty(num_blocks, HKV, PAGE, 2 * D, dtype=torch.bfloat16, device=device)
    else:
        raw = torch.empty(num_blocks, PAGE, HKV, 2 * D, dtype=torch.bfloat16, device=device)
    raw.normal_(generator=g)
    raw[..., :D] *= k_amp
    raw[..., D:] *= v_amp
    # a few outlier channels in K like real post-RoPE keys
    raw[..., 7] *= 6.0
    raw[..., 130] *= 4.0
    raw = raw.clamp(-448, 448).to(torch.float8_e4m3fn)
    k, v = raw.split(D, dim=-1)
    if layout == "NHD":
        k, v = k.transpose(1, 2), v.transpose(1, 2)
    return raw, k, v


def make_batch(seqs, num_blocks, seed=0, device="cuda"):
    """seqs: list of (q_len, kv_len) with kv_len >= q_len (kv_len - q_len = cached prefix).

    Returns the FlashInfer-wrapper-like metadata vLLM's glue reads:
    paged_kv_indices (absolute block ids, random), indptr, last_page_len, qo_indptr.
    """
    g = torch.Generator().manual_seed(seed)
    npages = [math.ceil(kv / PAGE) for _, kv in seqs]
    total = sum(npages)
    assert total <= num_blocks
    perm = torch.randperm(num_blocks, generator=g)[:total]
    indptr = [0]
    for n in npages:
        indptr.append(indptr[-1] + n)
    last = [kv - (n - 1) * PAGE for (_, kv), n in zip(seqs, npages)]
    qo = [0]
    for q, _ in seqs:
        qo.append(qo[-1] + q)
    return dict(
        indices=perm.to(device=device, dtype=torch.int32),
        indptr=torch.tensor(indptr, dtype=torch.int32, device=device),
        last=torch.tensor(last, dtype=torch.int32, device=device),
        qo_indptr=torch.tensor(qo, dtype=torch.int32, device=device),
        max_q=max(q for q, _ in seqs),
        max_kv=max(kv for _, kv in seqs),
        total_q=qo[-1],
    )


def make_query(total_q, seed=1, amp=2.5, device="cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.empty(total_q, HQ, D, dtype=torch.bfloat16, device=device).normal_(generator=g)
    q *= amp
    q[..., 3] *= 8.0
    return q


def baseline_meta(b):
    """Exactly what patch 01 builds once per step (compact-gather block table)."""
    idx, indptr = b["indices"], b["indptr"]
    max_pages = -(-b["max_kv"] // PAGE)
    bt = indptr[:-1, None] + torch.arange(max_pages, device=idx.device, dtype=indptr.dtype)[None, :]
    bt = bt.clamp_(max=max(idx.shape[0] - 1, 0)).to(torch.int32)
    seqlens = ((indptr[1:] - indptr[:-1] - 1) * PAGE + b["last"]).to(torch.int32)
    return idx, bt, seqlens, b["qo_indptr"].to(torch.int32)


def new_meta(b):
    from glue_fmha import absolute_block_table

    max_pages = -(-b["max_kv"] // PAGE)
    bt = absolute_block_table(b["indices"], b["indptr"], max_pages)
    seqlens = ((b["indptr"][1:] - b["indptr"][:-1] - 1) * PAGE + b["last"]).to(torch.int32)
    return bt, seqlens, b["qo_indptr"].to(torch.int32)


def fi_kernel(out_dtype=torch.bfloat16, kv_tile=128, q_tile=128, balanced=True):
    """The kernel patch 01 uses (FlashInfer's own compile + disk cache)."""
    from flashinfer.cute_dsl.attention.fmha.sm120 import compile_sm120_fmha_fp8_paged_kernel

    return compile_sm120_fmha_fp8_paged_kernel(
        in_dtype=torch.float8_e4m3fn, out_dtype=out_dtype, num_qo_heads=HQ,
        num_kv_heads=HKV, head_dim=D, is_causal=True, kv_tile=kv_tile, q_tile=q_tile,
        num_tokens_per_page=PAGE, device=torch.device("cuda"), with_lse=False,
        balanced_scheduler=balanced,
    )


def quant_q_baseline(query, q_scale):
    q = query if q_scale == 1.0 else query / q_scale
    return q.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def run_baseline(fn, query, k, v, out, meta, max_q, sm_scale, q_scale, k_scale, v_scale):
    """Body of patch-01 _sm120_fp8_paged_prefill (after meta is cached)."""
    from cutlass.cute.typing import Float32, Int32

    idx, bt, seqlens, cu_q = meta
    kc = k.index_select(0, idx)
    vc = v.index_select(0, idx)
    q8 = quant_q_baseline(query, q_scale)
    s = sm_scale * q_scale * k_scale
    fn(q8, kc, vc, out, None, Float32(s * LOG2E), Float32(v_scale), seqlens, cu_q, bt,
       None, Int32(max_q), False)
    return out


def run_new(fn, query, k, v, out, meta, max_q, sm_scale, q_scale, k_scale, v_scale):
    """Glue-free path: strided K/V views + absolute block table + BF16 Q in kernel."""
    from cutlass.cute.typing import Float32, Int32

    bt, seqlens, cu_q = meta
    s = sm_scale * q_scale * k_scale
    fn(query, k, v, out, None, Float32(s * LOG2E), Float32(v_scale), seqlens, cu_q, bt,
       None, Int32(max_q), False)
    return out


def ref_attention(query, k, v, b, sm_scale, q_scale, k_scale, v_scale):
    """FP32 reference on the dequantised FP8 operands (bottom-right causal)."""
    q8 = quant_q_baseline(query, q_scale).float() * q_scale
    outs = []
    idx = b["indices"].long()
    indptr = b["indptr"].tolist()
    qo = b["qo_indptr"].tolist()
    for i in range(len(qo) - 1):
        pages = idx[indptr[i]:indptr[i + 1]]
        kv_len = (indptr[i + 1] - indptr[i] - 1) * PAGE + int(b["last"][i])
        kk = k[pages].float().permute(0, 2, 1, 3).reshape(-1, HKV, D)[:kv_len] * k_scale
        vv = v[pages].float().permute(0, 2, 1, 3).reshape(-1, HKV, D)[:kv_len] * v_scale
        qq = q8[qo[i]:qo[i + 1]]
        lq = qq.shape[0]
        kk = kk.repeat_interleave(HQ // HKV, dim=1)
        vv = vv.repeat_interleave(HQ // HKV, dim=1)
        s = torch.einsum("qhd,khd->hqk", qq, kk) * sm_scale
        qpos = torch.arange(lq, device=s.device)[:, None] + (kv_len - lq)
        kpos = torch.arange(kv_len, device=s.device)[None, :]
        s = s.masked_fill(kpos > qpos, float("-inf"))
        p = torch.softmax(s, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", p, vv))
    return torch.cat(outs, 0)


def causal_flops(seqs):
    return sum(4 * HQ * D * (q * kv - q * q / 2) for q, kv in seqs)
