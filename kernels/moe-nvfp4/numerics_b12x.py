"""Why does the FlashInfer b12x MoE backend degrade quality in vLLM v0.32.0?

Compares, on one real layer with real (captured) MoE inputs and real routing, against an fp32 reference
(exactly dequantized weights, unquantized activations):
  cutlass        : FlashInfer CUTLASS fused MoE exactly as vLLM's FLASHINFER_CUTLASS backend calls it
  b12x_vllm      : B12xMoEWrapper exactly as vLLM's FlashInferB12xExperts calls it
                   (weight global scale baked into the e4m3 block scales, all activation global scales = 1)
  b12x_bake_only : vLLM's baked weight scales, but calibrated activation global scales
  b12x_act1_only : original weight scales, but activation global scales forced to 1 (vLLM's choice)
  b12x_fixed     : original e4m3 block scales + calibrated static activation global scales (same scales as
                   the CUTLASS path)
usage: numerics_b12x.py [layer ...]
"""
import sys, torch
from moe_common import *


_KEEP = []


def b12x_run(wr, x, ids, wts, w13, s13_sw, w2, s2_sw, w1_alpha, w2_alpha, fc2_scale, in_gs):
    """Fresh wrapper per variant and every tensor kept alive: B12xMoEWrapper caches weight views and the
    folded alpha by data_ptr, so recycled temporaries would otherwise silently reuse a stale variant."""
    from flashinfer.fused_moe import B12xMoEWrapper
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    wr = B12xMoEWrapper(num_experts=E, top_k=TOPK, hidden_size=H, intermediate_size=I,
                        use_cuda_graph=False, activation="silu")
    s13m = convert_sf_to_mma_layout(s13_sw.reshape(E * 2 * I, H // 16), m=2 * I, k=H, num_groups=E)
    s2m = convert_sf_to_mma_layout(s2_sw.reshape(E * H, I // 16), m=H, k=I, num_groups=E)
    _KEEP.append((wr, s13m, s2m, w1_alpha, w2_alpha, fc2_scale, in_gs, s13_sw, s2_sw))
    out = wr.run(x=x, w1_weight=w13, w1_weight_sf=s13m, w1_alpha=w1_alpha, fc2_input_scale=fc2_scale,
                 w2_weight=w2, w2_weight_sf=s2m, w2_alpha=w2_alpha, token_selected_experts=ids,
                 token_final_scales=wts, input_global_scale=in_gs)
    torch.cuda.synchronize()
    return out.clone()


def main():
    from flashinfer.fused_moe import B12xMoEWrapper
    layers = [int(a) for a in sys.argv[1:]] or [0, 19, 39]
    torch.manual_seed(0)
    for L in layers:
        raw = load_layer_raw(L)
        x = load_hidden(L)
        src = "captured"
        if x is None:
            src = "synthetic"
            x = (torch.randn(8192, H, device="cuda") * 0.6).to(torch.bfloat16)
        x = x[:4096].contiguous()
        wts, ids = route(x, raw["gate"])
        ref = reference(x, raw, ids, wts)
        prep = prep_cutlass(raw)
        res = {"cutlass": run_cutlass(prep, x, ids, wts).clone()}

        wr = B12xMoEWrapper(num_experts=E, top_k=TOPK, hidden_size=H, intermediate_size=I,
                            use_cuda_graph=False, activation="silu")
        ones = torch.ones(E, device="cuda")
        inv_w13 = (1.0 / raw["wgs13"]).view(E, 1, 1)
        inv_w2 = (1.0 / raw["wgs2"]).view(E, 1, 1)
        # vLLM bake: (fp8 swizzled scale * w_scale_2) -> fp8
        s13_baked = (prep["s13"].float() * inv_w13).to(torch.float8_e4m3fn)
        s2_baked = (prep["s2"].float() * inv_w2).to(torch.float8_e4m3fn)
        a1g, a2g = prep["a1_gscale"], prep["a2_gscale"]
        # launch folds w1_alpha *= input_global_scale; the dequant factor must be g1 = 1/(a1g*wgs13),
        # so pass g1/a1g.  FC2 is not folded: w2_alpha = g2, fc2_input_scale = a2g.
        res["b12x_vllm"] = b12x_run(wr, x, ids, wts, prep["w13"], s13_baked, prep["w2"], s2_baked,
                                    ones, ones, ones, None)
        inv13 = (1.0 / raw["wgs13"]).contiguous(); inv2 = (1.0 / raw["wgs2"]).contiguous()
        # unit activation global scales (b12x dynamic per-16 block scaling only), fp32 weight alphas
        res["b12x_act1"] = b12x_run(wr, x, ids, wts, prep["w13"], prep["s13"], prep["w2"], prep["s2"],
                                    inv13, inv2, ones, ones)
        # calibrated static activation scales, passed the way b12x actually interprets them: as
        # reciprocal (dequant-style) scales a13 = 1/a1_gscale, a2 = 1/a2_gscale; launch folds
        # w1_alpha *= input_global_scale, FC2 is not folded.
        a13v = (1.0 / a1g).contiguous(); a2v = (1.0 / a2g).contiguous()
        res["b12x_static"] = b12x_run(wr, x, ids, wts, prep["w13"], prep["s13"], prep["w2"], prep["s2"],
                                      inv13, (inv2 * a2v).contiguous(), a2v, a13v)
        # probes that show the reciprocal semantics: passing a quantization multiplier c as the FC2
        # input scale (with w2_alpha = 1/(wgs2*c)) shrinks the output by 1/c^2
        for c in (2.0, 8.0):
            cg = torch.full((E,), c, device="cuda")
            res[f"probe_fc2_mult{c:g}"] = b12x_run(wr, x, ids, wts, prep["w13"], prep["s13"], prep["w2"], prep["s2"],
                                                   inv13, (inv2 / c).contiguous(), cg, ones)
        # what one would pass from the docstring ("global scale for input quantization" = a1_gscale)
        res["probe_static_as_mult"] = b12x_run(wr, x, ids, wts, prep["w13"], prep["s13"], prep["w2"], prep["s2"],
                                               (prep["g1"] / a1g).contiguous(), prep["g2"], a2g, a1g)
        print(f"== layer {L} ({src} inputs, {x.shape[0]} tokens; a1_gscale={a1g[0].item():.1f} "
              f"a2_gscale={a2g[0].item():.1f}; |x|max={x.abs().max().item():.2f})")
        for k, v in res.items():
            r = rel_err(v, ref)
            c = rel_err(v, res["cutlass"])
            print(f"  {k:15s} vs fp32 ref: rel_l2 {r['rel_l2']:.4f} cos {r['cos']:.6f} max_abs {r['max_abs']:.4f}"
                  f"   | vs cutlass: rel_l2 {c['rel_l2']:.4f}")
        del raw, prep
        _KEEP.clear()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
