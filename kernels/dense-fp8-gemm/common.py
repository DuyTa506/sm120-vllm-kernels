"""Shared helpers: real FP8 weights of Qwen3.6-35B-A3B-NVFP4-Fast, realistic activations, timing."""
import glob
import json
import os
import statistics

import torch

_PAT = "hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/*/"
SNAP = (glob.glob("/hf/" + _PAT) + glob.glob("/root/.cache/huggingface/" + _PAT))[0]
P = "model.language_model.layers."
GDN_LAYERS = [i for i in range(40) if i % 4 != 3]
ATTN_LAYERS = [i for i in range(40) if i % 4 == 3]

# name -> (N, K, calls per forward, layers it lives in)
SHAPES = {
    "gdn_qkvz": (12288, 2048, 30),   # in_proj_qkv (8192) + in_proj_z (4096) merged by vLLM
    "gdn_out": (2048, 4096, 30),
    "attn_qkv": (9216, 2048, 10),    # q (8192 incl. output gate) + k 512 + v 512
    "attn_o": (2048, 4096, 10),
    "lm_head": (248320, 2048, 1),
}

_index = None


def _file_of(name):
    global _index
    if _index is None:
        _index = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]
    return os.path.join(SNAP, _index[name])


def _load(names):
    from safetensors import safe_open
    out = {}
    by_file = {}
    for n in names:
        by_file.setdefault(_file_of(n), []).append(n)
    for f, ns in by_file.items():
        with safe_open(f, "pt", device="cpu") as fh:
            for n in ns:
                out[n] = fh.get_tensor(n)
    return out


def load_weight(shape_name, layer):
    """Return (w [N,K] fp8 cuda row-major, scale [N] fp32 cuda) exactly as vLLM merges them."""
    if shape_name == "lm_head":
        parts = ["lm_head"]
    elif shape_name == "gdn_qkvz":
        parts = [f"{P}{layer}.linear_attn.in_proj_qkv", f"{P}{layer}.linear_attn.in_proj_z"]
    elif shape_name == "gdn_out":
        parts = [f"{P}{layer}.linear_attn.out_proj"]
    elif shape_name == "attn_qkv":
        parts = [f"{P}{layer}.self_attn.{x}_proj" for x in "qkv"]
    elif shape_name == "attn_o":
        parts = [f"{P}{layer}.self_attn.o_proj"]
    t = _load([p + s for p in parts for s in (".weight", ".weight_scale")])
    w = torch.cat([t[p + ".weight"] for p in parts], 0)
    s = torch.cat([t[p + ".weight_scale"].float().reshape(-1) for p in parts], 0)
    return w.cuda(), s.cuda()


def layers_for(shape_name):
    if shape_name == "lm_head":
        return [None]
    return GDN_LAYERS if shape_name.startswith("gdn") else ATTN_LAYERS


def make_act(M, K, seed=0, device="cuda"):
    """bf16 activations with LLM-like per-channel outlier structure (lognormal channel scale,
    a few large outlier channels), then per-token dynamic FP8 quant like vLLM."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ch = torch.exp(torch.randn(K, generator=g) * 0.6)
    idx = torch.randperm(K, generator=g)[: max(1, K // 256)]
    ch[idx] *= 20.0
    x = (torch.randn(M, K, generator=g) * ch).to(torch.bfloat16)
    return x.to(device)


def quant_per_token(x):
    from vllm import _custom_ops as ops
    return ops.scaled_fp8_quant(x, use_per_token_if_dynamic=True)


def ref_mm(a_q, a_s, w, w_s, out_dtype=torch.bfloat16):
    """fp32 reference of the scaled mm (exact products, fp32/fp64-ish accumulation)."""
    acc = a_q.float() @ w.float().t()
    return (acc * a_s.view(-1, 1) * w_s.view(1, -1)).to(out_dtype), acc * a_s.view(-1, 1) * w_s.view(1, -1)


def bench_graph(fn_list, iters=30, warmup=3):
    """Capture all callables of fn_list into one CUDA graph (in order) and time replays.
    Returns median time per callable in microseconds."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fn_list:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fn_list:
            f()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        e1.synchronize()
        ts.append(e0.elapsed_time(e1) * 1e3 / len(fn_list))
    del g
    return statistics.median(ts)


def err_stats(out, ref32):
    d = (out.float() - ref32).abs()
    rel = d.max().item() / max(ref32.abs().max().item(), 1e-30)
    return d.max().item(), rel
