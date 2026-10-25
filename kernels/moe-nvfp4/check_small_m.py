"""Small-M (decode) correctness: CUTLASS and b12x (vLLM-baked / fixed static scales) vs the fp32 reference,
for M = 1..256 tokens (real captured inputs, real routing), both eager and inside CUDA graphs.
usage: check_small_m.py [layer]"""
import sys, torch
from moe_common import *


def main():
    from flashinfer.fused_moe import B12xMoEWrapper
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 19
    raw = load_layer_raw(L); prep = prep_cutlass(raw)
    xall = load_hidden(L)
    s13m = convert_sf_to_mma_layout(prep["s13"].reshape(E * 2 * I, H // 16), m=2 * I, k=H, num_groups=E)
    s2m = convert_sf_to_mma_layout(prep["s2"].reshape(E * H, I // 16), m=H, k=I, num_groups=E)
    a13v = (1.0 / prep["a1_gscale"]).contiguous(); a2v = (1.0 / prep["a2_gscale"]).contiguous()
    w1a = (1.0 / raw["wgs13"]).contiguous(); w2a = ((1.0 / raw["wgs2"]) * a2v).contiguous()
    for graph_mode in (False, True):
        wr = B12xMoEWrapper(num_experts=E, top_k=TOPK, hidden_size=H, intermediate_size=I,
                            use_cuda_graph=graph_mode, max_num_tokens=16384, activation="silu")
        print(f"-- layer {L}, B12xMoEWrapper(use_cuda_graph={graph_mode})")
        for M in (1, 2, 3, 4, 6, 8, 12, 16, 24, 48, 96, 192, 256):
            torch.manual_seed(M)
            x = xall[torch.randperm(xall.shape[0], device="cuda")[:M]].contiguous()
            wts, ids = route(x, raw["gate"])
            ref = reference(x, raw, ids, wts)
            yc = run_cutlass(prep, x, ids, wts).clone()
            fb = lambda: wr.run(x=x, w1_weight=prep["w13"], w1_weight_sf=s13m, w1_alpha=w1a, fc2_input_scale=a2v,
                                w2_weight=prep["w2"], w2_weight_sf=s2m, w2_alpha=w2a,
                                token_selected_experts=ids, token_final_scales=wts, input_global_scale=a13v)
            if graph_mode:
                fb(); torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    fb()
                torch.cuda.current_stream().wait_stream(s)
                with torch.cuda.graph(g):
                    o = fb()
                g.replay(); torch.cuda.synchronize()
                yb = o.clone()
            else:
                yb = fb().clone()
            rc, rb, rbc = rel_err(yc, ref), rel_err(yb, ref), rel_err(yb, yc)
            print(f"  M={M:4d}: cutlass vs ref {rc['rel_l2']:.4f} | b12x vs ref {rb['rel_l2']:.4f} | b12x vs cutlass "
                  f"{rbc['rel_l2']:.4f} | |b12x| {yb.float().norm().item():.3e} |ref| {ref.norm().item():.3e}")


if __name__ == "__main__":
    main()
