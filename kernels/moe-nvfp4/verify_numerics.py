"""VERIFIER numerics: b12x (fixed static scales, as the hybrid patch calls it) vs CUTLASS vs fp32 reference.
Checks not covered by numerics_b12x.py / check_small_m.py:
  1. full 8192-token batch (the prefill size the speedup is claimed for), not 4096
  2. per-token batch invariance: rows of a 2048 batch vs the same rows inside an 8192 batch
  3. hybrid threshold discontinuity: same 1023 tokens through CUTLASS (M=1023) vs b12x (M=1024)
  4. padded/zero rows (CUDA-graph padding) do not produce NaN/Inf nor disturb other rows
  5. expert id -1 rows (masked to expert 0 / weight 0 as the patch does) vs CUTLASS with -1
  6. run-to-run determinism of each kernel
  7. M in {1000, 1024, 1025, 1100, 1536} odd sizes around the threshold
usage: verify_numerics.py [layer ...]
"""
import sys, torch
from moe_common import *


def main():
    from flashinfer.fused_moe import B12xMoEWrapper
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    layers = [int(a) for a in sys.argv[1:]] or [19, 39]
    for L in layers:
        raw = load_layer_raw(L); prep = prep_cutlass(raw)
        xall = load_hidden(L)
        s13m = convert_sf_to_mma_layout(prep["s13"].reshape(E * 2 * I, H // 16), m=2 * I, k=H, num_groups=E)
        s2m = convert_sf_to_mma_layout(prep["s2"].reshape(E * H, I // 16), m=H, k=I, num_groups=E)
        a13v = (1.0 / prep["a1_gscale"]).contiguous(); a2v = (1.0 / prep["a2_gscale"]).contiguous()
        w1a = (1.0 / raw["wgs13"]).contiguous(); w2a = ((1.0 / raw["wgs2"]) * a2v).contiguous()
        wr = B12xMoEWrapper(num_experts=E, top_k=TOPK, hidden_size=H, intermediate_size=I,
                            use_cuda_graph=True, max_num_tokens=16384, activation="silu")

        def b12x(x, ids, wts):
            skip = ids < 0
            ids2 = ids.masked_fill(skip, 0); wts2 = wts.masked_fill(skip, 0.0)
            y = wr.run(x=x, w1_weight=prep["w13"], w1_weight_sf=s13m, w1_alpha=w1a, fc2_input_scale=a2v,
                       w2_weight=prep["w2"], w2_weight_sf=s2m, w2_alpha=w2a,
                       token_selected_experts=ids2.to(torch.int32), token_final_scales=wts2,
                       input_global_scale=a13v)
            torch.cuda.synchronize()
            return y.clone()

        def cut(x, ids, wts):
            y = run_cutlass(prep, x, ids, wts).clone(); torch.cuda.synchronize(); return y

        print(f"== layer {L}", flush=True)
        x = xall[:8192].contiguous()
        wts, ids = route(x, raw["gate"])
        ref = reference(x, raw, ids, wts)
        yc, yb = cut(x, ids, wts), b12x(x, ids, wts)
        rc, rb, rbc = rel_err(yc, ref), rel_err(yb, ref), rel_err(yb, yc)
        print(f" [1] M=8192: cutlass vs ref rel_l2 {rc['rel_l2']:.4f} max_abs {rc['max_abs']:.4f} | b12x vs ref "
              f"{rb['rel_l2']:.4f} max_abs {rb['max_abs']:.4f} | b12x vs cutlass {rbc['rel_l2']:.4f} max_abs "
              f"{rbc['max_abs']:.4f} (ref absmax {rc['ref_absmax']:.3f}); finite b12x {torch.isfinite(yb).all().item()}")
        # per-row error distribution b12x vs cutlass
        dr = ((yb.float() - yc.float()).norm(dim=1) / yc.float().norm(dim=1).clamp_min(1e-6))
        q = torch.quantile(dr, torch.tensor([0.5, 0.99, 1.0], device="cuda")).tolist()
        print(f"     per-token rel err b12x vs cutlass p50 {q[0]:.4f} p99 {q[1]:.4f} max {q[2]:.4f}")
        # 2 batch invariance
        yb2 = b12x(x[:2048].contiguous(), ids[:2048].contiguous(), wts[:2048].contiguous())
        yc2 = cut(x[:2048].contiguous(), ids[:2048].contiguous(), wts[:2048].contiguous())
        print(f" [2] batch invariance rows[:2048] (2048-batch vs 8192-batch): b12x max_abs "
              f"{(yb2.float() - yb[:2048].float()).abs().max().item():.3e}, cutlass max_abs "
              f"{(yc2.float() - yc[:2048].float()).abs().max().item():.3e}")
        # 3 threshold discontinuity
        x3, i3, w3 = x[:1024].contiguous(), ids[:1024].contiguous(), wts[:1024].contiguous()
        yc3 = cut(x3[:1023].contiguous(), i3[:1023].contiguous(), w3[:1023].contiguous())
        yb3 = b12x(x3, i3, w3)
        r3 = rel_err(yb3[:1023], yc3)
        print(f" [3] hybrid switch: same 1023 tokens via CUTLASS (M=1023) vs b12x (M=1024): rel_l2 {r3['rel_l2']:.4f}"
              f" max_abs {r3['max_abs']:.4f}")
        # 4 zero/padded rows
        x4 = x[:2048].clone(); x4[-64:] = 0
        w4, i4 = route(x4, raw["gate"])
        yb4 = b12x(x4, i4, w4); yc4 = cut(x4, i4, w4)
        print(f" [4] 64 zero rows: b12x finite {torch.isfinite(yb4).all().item()} zero-row absmax "
              f"{yb4[-64:].float().abs().max().item():.3e}; other rows vs unpadded b12x max_abs "
              f"{(yb4[:-64].float() - yb2[:-64].float()).abs().max().item():.3e}; cutlass finite "
              f"{torch.isfinite(yc4).all().item()}")
        # 5 negative ids
        i5 = ids[:2048].clone(); w5 = wts[:2048].clone()
        i5[::7, -1] = -1
        yb5 = b12x(x[:2048].contiguous(), i5, w5)
        w5z = w5.masked_fill(i5 < 0, 0.0); i5z = i5.masked_fill(i5 < 0, 0)
        yc5 = cut(x[:2048].contiguous(), i5z, w5z)
        print(f" [5] expert id -1 on every 7th token: b12x(masked) vs cutlass(masked, same) rel_l2 "
              f"{rel_err(yb5, yc5)['rel_l2']:.4f}; finite {torch.isfinite(yb5).all().item()}")
        # 6 determinism
        ybr, ycr = b12x(x, ids, wts), cut(x, ids, wts)
        print(f" [6] rerun bitwise equal: b12x {torch.equal(ybr, yb)} cutlass {torch.equal(ycr, yc)} "
              f"(max_abs {(ybr.float() - yb.float()).abs().max().item():.2e} / {(ycr.float() - yc.float()).abs().max().item():.2e})")
        # 7 odd sizes
        for M in (1000, 1024, 1025, 1100, 1536, 3000):
            xs, i7, w7 = x[:M].contiguous(), ids[:M].contiguous(), wts[:M].contiguous()
            rr = reference(xs, raw, i7, w7)
            a, b = cut(xs, i7, w7), b12x(xs, i7, w7)
            print(f" [7] M={M}: cutlass vs ref {rel_err(a, rr)['rel_l2']:.4f} b12x vs ref {rel_err(b, rr)['rel_l2']:.4f}"
                  f" b12x vs cutlass {rel_err(b, a)['rel_l2']:.4f}", flush=True)
        del wr, raw, prep, ref
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
