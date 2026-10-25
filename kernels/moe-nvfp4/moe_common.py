"""Shared helpers: load one real NVFP4 MoE layer of unsloth/Qwen3.6-35B-A3B-NVFP4-Fast, prepare it the way
vLLM v0.32.0 does for the FlashInfer CUTLASS backend (compressed-tensors W4A4 NVFP4 path), build a
fp32 dequantized reference, real or synthetic routing, and run the CUTLASS / b12x kernels.
"""
import glob, json, os
import torch
from safetensors import safe_open

E, TOPK, H, I = 256, 8, 2048, 512
HERE = os.path.dirname(os.path.abspath(__file__))
_SNAP = None
_HANDLES = {}


def snap():
    global _SNAP
    if _SNAP is None:
        _SNAP = glob.glob("/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/*/")[0]
    return _SNAP


def get_tensor(name):
    wmap = json.load(open(snap() + "model.safetensors.index.json"))["weight_map"] if not _HANDLES.get("_map") else _HANDLES["_map"]
    _HANDLES["_map"] = wmap
    f = wmap[name]
    if f not in _HANDLES:
        _HANDLES[f] = safe_open(snap() + f, "pt")
    return _HANDLES[f].get_tensor(name)


def load_layer_raw(L, dev="cuda"):
    """Checkpoint tensors stacked over experts, in vLLM's w13 = [w1(gate); w3(up)] order."""
    p = f"model.language_model.layers.{L}.mlp."
    w13, s13, w2, s2, gs13, gs2, igs13, igs2 = [], [], [], [], [], [], [], []
    for e in range(E):
        q = f"{p}experts.{e}."
        g, u, d = (get_tensor(q + n + ".weight_packed") for n in ("gate_proj", "up_proj", "down_proj"))
        sg, su, sd = (get_tensor(q + n + ".weight_scale") for n in ("gate_proj", "up_proj", "down_proj"))
        w13.append(torch.cat([g, u], 0)); s13.append(torch.cat([sg, su], 0))
        w2.append(d); s2.append(sd)
        gs13.append(get_tensor(q + "gate_proj.weight_global_scale").float().reshape(1))
        gs2.append(get_tensor(q + "down_proj.weight_global_scale").float().reshape(1))
        igs13.append(get_tensor(q + "gate_proj.input_global_scale").float().reshape(1))
        igs2.append(get_tensor(q + "down_proj.input_global_scale").float().reshape(1))
    t = lambda xs: torch.stack(xs).to(dev)
    return dict(w13=t(w13), s13=t(s13), w2=t(w2), s2=t(s2), wgs13=t(gs13).view(E), wgs2=t(gs2).view(E),
                igs13=t(igs13).view(E), igs2=t(igs2).view(E),
                gate=get_tensor(p + "gate.weight").to(dev),
                shared_gate=get_tensor(p + "shared_expert_gate.weight").to(dev))


_FP4_LUT = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


def dequant(packed, scale, wgs):
    """packed [E, N, K/2] uint8, scale [E, N, K/16] e4m3 (linear layout), wgs [E] -> fp32 [E, N, K]."""
    lut = _FP4_LUT.to(packed.device)
    lo, hi = lut[(packed & 0xF).long()], lut[(packed >> 4).long()]
    w = torch.stack([lo, hi], -1).flatten(-2)
    Ee, N, K = w.shape
    w = w.view(Ee, N, K // 16, 16) * scale.float().unsqueeze(-1)
    return (w.view(Ee, N, K) / wgs.view(Ee, 1, 1))


def route(x, gate_w, topk=TOPK):
    """Qwen3.5-MoE router: softmax over 256 experts, top-8, renormalize (norm_topk_prob=True)."""
    logits = torch.nn.functional.linear(x, gate_w).float()
    p = torch.softmax(logits, -1)
    w, ids = p.topk(topk, -1)
    w = w / w.sum(-1, keepdim=True)
    return w.float(), ids.to(torch.int32)


def reference(x, raw, ids, wts):
    """fp32 reference with exact dequantized weights and unquantized activations."""
    W13 = dequant(raw["w13"], raw["s13"], raw["wgs13"])   # [E, 2I, H] rows: gate | up
    W2 = dequant(raw["w2"], raw["s2"], raw["wgs2"])        # [E, H, I]
    xf = x.float()
    out = torch.zeros(x.shape[0], H, device=x.device)
    flat = ids.view(-1).long()
    tok = torch.arange(x.shape[0], device=x.device).repeat_interleave(ids.shape[1])
    wf = wts.view(-1)
    order = torch.argsort(flat)
    flat, tok, wf = flat[order], tok[order], wf[order]
    counts = torch.bincount(flat, minlength=E).tolist()
    s = 0
    for e, c in enumerate(counts):
        if c == 0:
            continue
        t = tok[s:s + c]
        h = xf[t] @ W13[e].T
        a = torch.nn.functional.silu(h[:, :I]) * h[:, I:]
        out.index_add_(0, t, (a @ W2[e].T) * wf[s:s + c, None])
        s += c
    return out


def prep_cutlass(raw):
    """Mirror vLLM CompressedTensorsW4A4Nvfp4MoEMethod + prepare_nvfp4_moe_layer_for_fi_or_cutlass
    + FlashInferExperts.process_weights_after_loading for the FLASHINFER_CUTLASS backend."""
    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import reorder_w1w3_to_w3w1
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale
    w13, s13 = reorder_w1w3_to_w3w1(raw["w13"].clone(), raw["s13"].clone())
    a13 = (1.0 / raw["igs13"]).max()     # global input scale shared by all experts
    a2 = (1.0 / raw["igs2"]).max()
    g1 = (1.0 / raw["wgs13"]) * a13      # w13_weight_scale_2 * w13_input_scale
    g2 = (1.0 / raw["wgs2"]) * a2
    return dict(w13=w13.contiguous(), s13=swizzle_blockscale(s13), w2=raw["w2"].contiguous(),
                s2=swizzle_blockscale(raw["s2"]), a1_gscale=(1.0 / a13).repeat(E).contiguous(),
                a2_gscale=(1.0 / a2).repeat(E).contiguous(), g1=g1.contiguous(), g2=g2.contiguous(),
                a13=a13, a2=a2)


def quant_input(x, a1_gscale):
    from vllm import _custom_ops as ops
    return ops.scaled_fp4_quant(x, a1_gscale[:1].contiguous() if a1_gscale.numel() > 1 else a1_gscale)


def run_cutlass(prep, x, ids, wts, out=None, profile_ids=None, xq=None):
    from flashinfer.fused_moe import cutlass_fused_moe
    if xq is None:
        xq = quant_input(x, prep["a1_gscale"])
    xq_, xsf = xq
    if out is None:
        out = torch.empty(x.shape[0], H, device=x.device, dtype=torch.bfloat16)
    cutlass_fused_moe(input=xq_, token_selected_experts=ids, token_final_scales=wts,
                      fc1_expert_weights=prep["w13"].view(torch.long), fc2_expert_weights=prep["w2"].view(torch.long),
                      output_dtype=torch.bfloat16, output=out,
                      quant_scales=[prep["a1_gscale"], prep["s13"].view(torch.int32), prep["g1"],
                                    prep["a2_gscale"], prep["s2"].view(torch.int32), prep["g2"]],
                      input_sf=xsf, profile_ids=profile_ids, tune_max_num_tokens=16384)
    return out


def rel_err(y, ref):
    y, ref = y.float(), ref.float()
    d = y - ref
    return dict(rel_l2=(d.norm() / ref.norm()).item(), max_abs=d.abs().max().item(),
                ref_absmax=ref.abs().max().item(),
                cos=torch.nn.functional.cosine_similarity(y.flatten(), ref.flatten(), 0).item())


def load_hidden(L, n=None, dev="cuda"):
    """Real captured MoE inputs (data/moe_inputs.pt) if present, else None."""
    p = os.path.join(HERE, "data/moe_inputs.pt")
    if not os.path.exists(p):
        return None
    d = torch.load(p)
    if L not in d["hidden"]:
        return None
    x = d["hidden"][L].to(dev)
    return x[:n] if n else x
