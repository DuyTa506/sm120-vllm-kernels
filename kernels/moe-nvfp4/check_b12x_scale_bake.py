"""CPU-only check: how much precision does vLLM's FlashInferB12xExperts weight-scale "bake-in" lose?

vLLM (flashinfer_b12x_moe.py, process_weights_after_loading) does
    w13_weight_scale = (w13_weight_scale.float() * w13_weight_scale_2).to(float8_e4m3fn)
i.e. it multiplies every e4m3 block scale by the per-expert global weight scale (1/weight_global_scale,
~1e-4) and re-rounds to e4m3. The products land mostly in / below the e4m3 subnormal range (< 2^-6).
This script measures the relative error of the effective dequant block scale on real checkpoint weights.
"""
import glob, json, sys, torch
from safetensors import safe_open

snap = glob.glob("/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/*/")[0]
wmap = json.load(open(snap + "model.safetensors.index.json"))["weight_map"]
layers = [int(x) for x in sys.argv[1:]] or [0, 20, 39]
handles = {}
def get(name):
    f = wmap[name]
    if f not in handles:
        handles[f] = safe_open(snap + f, "pt")
    return handles[f].get_tensor(name)

for L in layers:
    for proj in ["gate_proj", "up_proj", "down_proj"]:
        errs, zero_frac, sub_frac = [], [], []
        for e in range(0, 256, 8):
            p = f"model.language_model.layers.{L}.mlp.experts.{e}.{proj}."
            s = get(p + "weight_scale").float()               # e4m3 block scales (as float)
            gs = get(p + "weight_global_scale").float()       # compressed-tensors: 448*6/amax
            exact = s / gs                                    # effective dequant scale (what CUTLASS uses)
            baked = exact.to(torch.float8_e4m3fn).float()      # what vLLM b12x path stores
            nz = exact > 0
            rel = ((baked - exact).abs() / exact.clamp_min(1e-30))[nz]
            errs.append(rel)
            zero_frac.append(((baked == 0) & nz).float().mean().item())
            sub_frac.append(((exact < 2**-6) & nz).float().mean().item())
        rel = torch.cat(errs)
        print(f"layer {L:2d} {proj:9s}: gs~{gs.item():8.1f}  block-scale rel err: mean {rel.mean():.4f} "
              f"p50 {rel.median():.4f} p99 {rel.quantile(0.99):.4f} max {rel.max():.3f} | "
              f"subnormal {sum(sub_frac)/len(sub_frac)*100:.1f}%  flushed-to-0 {sum(zero_frac)/len(zero_frac)*100:.2f}%")
    for proj in ["gate_proj", "down_proj"]:
        p = f"model.language_model.layers.{L}.mlp.experts.0.{proj}."
        print(f"   {proj} input_global_scale (expert 0) = {get(p+'input_global_scale').float().item():.2f}")
