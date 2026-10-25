"""Shared helpers: realistic GDN layer-0 inputs from the real checkpoint, and the
reference (existing vLLM) two-kernel path."""

from __future__ import annotations

import glob
import json
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SNAP = glob.glob(
    "/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/*/"
)
H, HV, K, V = 16, 32, 128, 128
DIM = 2 * H * K + HV * V  # 8192
QKVZ = DIM + HV * V  # 12288: mixed_qkv is a column slice of the in_proj_qkvz output
DEV = os.environ.get("DEV", "cuda")  # DEV=cpu + TRITON_INTERPRET=1: GPU-free check
# the Triton interpreter mishandles bf16 arithmetic, so the CPU check runs the activations in fp32
ACT = torch.float32 if DEV == "cpu" else torch.bfloat16
if DEV == "cpu":  # vLLM swaps in a Triton placeholder when no GPU driver is present
    import types

    import triton
    import triton.language as tl

    _tu = types.ModuleType("vllm.triton_utils")
    _tu.triton, _tu.tl, _tu.HAS_TRITON = triton, tl, True
    sys.modules["vllm.triton_utils"] = _tu


def _load(names):
    from safetensors import safe_open

    snap = SNAP[0]
    wm = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
    out = {}
    for n in names:
        with safe_open(snap + wm[n], framework="pt", device="cpu") as f:
            out[n] = f.get_tensor(n)
    return out


_cache = {}


def layer_params(layer=0, dev=DEV):
    """conv1d weight/bias, A_log, dt_bias of a GDN layer and what is needed to
    project real embeddings into mixed_qkv / a / b."""
    if layer in _cache:
        return _cache[layer]
    p = f"model.language_model.layers.{layer}."
    la = p + "linear_attn."
    names = [
        "model.language_model.embed_tokens.weight",
        p + "input_layernorm.weight",
        la + "conv1d.weight",
        la + "A_log",
        la + "dt_bias",
        la + "in_proj_qkv.weight",
        la + "in_proj_qkv.weight_scale",
        la + "in_proj_a.weight",
        la + "in_proj_b.weight",
    ]
    t = _load(names)
    conv_w = t[la + "conv1d.weight"].to(dev)
    P = SimpleNamespace(
        embed=t[names[0]],  # keep on CPU (1 GB)
        ln_w=t[names[1]].to(dev),
        conv_w=conv_w.view(conv_w.shape[0], conv_w.shape[-1]).to(ACT).contiguous(),
        A_log=t[la + "A_log"].to(dev),
        dt_bias=t[la + "dt_bias"].to(dev),
        w_qkv=t[la + "in_proj_qkv.weight"].to(dev),
        s_qkv=t[la + "in_proj_qkv.weight_scale"].to(dev),
        w_a=t[la + "in_proj_a.weight"].to(dev),
        w_b=t[la + "in_proj_b.weight"].to(dev),
    )
    _cache[layer] = P
    return P


def token_ids(n, seed=0):
    """Real text tokens (vLLM source code tokenized with the model tokenizer)."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(SNAP[0] + "tokenizer.json")
    src = ""
    root = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/"
    for f in sorted(os.listdir(root)):
        if f.endswith(".py"):
            src += open(root + f).read()
        if len(src) > 8 * n:
            break
    ids = tok.encode(src).ids
    g = torch.Generator().manual_seed(seed)
    start = int(torch.randint(0, max(1, len(ids) - n), (1,), generator=g))
    return torch.tensor(ids[start : start + n], dtype=torch.long)


@torch.no_grad()
def make_inputs(T, layer=0, seed=0, dev=DEV):
    """mixed_qkv [T, 8192] (as a strided slice of a [T, 12288] buffer), a, b [T, 32]."""
    P = layer_params(layer, dev)
    ids = token_ids(T, seed)
    h = P.embed[ids].to(dev).float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + P.ln_w.float())
    h = h.to(torch.bfloat16)
    w = P.w_qkv.to(torch.bfloat16) * P.s_qkv.view(-1, 1).to(torch.bfloat16)
    qkvz = torch.zeros(T, QKVZ, dtype=ACT, device=dev)
    qkvz[:, :DIM] = (h @ w.t()).to(ACT)
    a = (h @ P.w_a.t()).contiguous()
    b = (h @ P.w_b.t()).contiguous()
    return qkvz[:, :DIM], a, b


def conv_meta(query_start_loc_cpu, dev=DEV):
    import vllm.v1.attention.backends.utils as U
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    if dev == "cpu":  # no pinned memory without a driver
        U.np_to_pinned_tensor = lambda a: torch.from_numpy(a)
        U.PIN_MEMORY = False

    nums_dict, batch_ptr, off = compute_causal_conv1d_metadata(
        query_start_loc_cpu, device=torch.device(dev)
    )
    return SimpleNamespace(nums_dict=nums_dict, batch_ptr=batch_ptr, token_chunk_offset_ptr=off)


def post_conv_ref():
    if DEV == "cpu":  # load the file directly (the fla package __init__ needs a GPU triton)
        import importlib.util

        f = ("/usr/local/lib/python3.12/dist-packages/vllm/third_party/flash_linear_attention/"
             "ops/fused_gdn_prefill_post_conv.py")
        spec = importlib.util.spec_from_file_location("_post_conv_ref", f)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m.fused_post_conv_prep
    from vllm.third_party.flash_linear_attention.ops.fused_gdn_prefill_post_conv import (
        fused_post_conv_prep,
    )

    return fused_post_conv_prep


def _conv_direct(x, w, cs, qsl, ci, has_init, block_idx_first_scheduled_token=None,
                 block_idx_last_scheduled_token=None, initial_state_idx=None,
                 num_computed_tokens=None, block_size_to_align=0):
    """Same launch as causal_conv1d_fn, with a precomputed grid (the Triton
    interpreter cannot run causal_conv1d_fn's grid lambda that swaps arguments)."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import _causal_conv1d_fwd_kernel

    from fused_conv_post_conv import _program_list

    xt = x.transpose(0, 1)
    out = torch.empty_like(xt)
    tot, bp, op = _program_list(qsl, x.device)
    bm = 8
    align = block_size_to_align if block_size_to_align else bm
    _causal_conv1d_fwd_kernel[(tot, DIM // 256)](
        xt, w, None, cs, ci, has_init, qsl, bp, op,
        block_idx_first_scheduled_token, block_idx_last_scheduled_token, initial_state_idx,
        num_computed_tokens, out, DIM, cs.size(0),
        xt.stride(0), xt.stride(1), w.stride(0), w.stride(1),
        cs.stride(0), cs.stride(1), cs.stride(2), ci.stride(0),
        out.stride(0), out.stride(1), align // bm, -1, 0,
        HAS_BIAS=False, KERNEL_WIDTH=4, SILU_ACTIVATION=True,
        IS_APC_ENABLED=block_idx_last_scheduled_token is not None, HAS_NULL_BLOCK=True,
        NP2_STATELEN=4, BLOCK_M=bm, BLOCK_N=256, launch_pdl=False,
    )
    return out.transpose(0, 1)


def reference(
    x, conv_w, conv_states, qsl, cache_indices, has_init, a, b, A_log, dt_bias,
    meta=None, output_g_exp=True, num_raw_tokens=0, apc=None,
):
    """Existing vLLM path: causal_conv1d_fn + fused_post_conv_prep (on prefill rows)."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    fused_post_conv_prep = post_conv_ref()
    apc = apc or {}
    if DEV == "cpu":
        out = _conv_direct(x, conv_w, conv_states, qsl, cache_indices, has_init, **apc)
    else:
        out = causal_conv1d_fn(
            x.transpose(0, 1), conv_w, None, activation="silu", conv_states=conv_states,
            has_initial_state=has_init, cache_indices=cache_indices, query_start_loc=qsl,
            metadata=meta, **apc,
        ).transpose(0, 1)
    raw = out[:num_raw_tokens]
    q, k, v, g, beta = fused_post_conv_prep(
        out[num_raw_tokens:], a[num_raw_tokens:], b[num_raw_tokens:], A_log, dt_bias,
        num_k_heads=H, head_k_dim=K, head_v_dim=V, apply_l2norm=True,
        output_g_exp=output_g_exp,
    )
    return q, k, v, g, beta, raw


def make_conv_states(num_lines, layout="SD", dev=DEV, seed=1):
    """vLLM conv cache. SD: per block (state_len, dim) -> kernels get the transpose."""
    gen = torch.Generator(device=dev).manual_seed(seed)
    if layout == "SD":
        base = torch.randn(num_lines, 3, DIM, device=dev, generator=gen).to(ACT)
        return base, base.transpose(-1, -2)
    base = torch.randn(num_lines, DIM, 3, device=dev, generator=gen).to(ACT)
    return base, base
