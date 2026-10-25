"""Correctness of the candidate kernels against vLLM's cutlass_scaled_mm and an fp32 reference, on real
weights of several layers (all 5 dense FP8 shapes) and LLM-like activations, over many M including
odd sizes (decode M = 3 * num_seqs with MTP-2, ragged prefill chunks).

Reports per (shape, M): bitwise-equal fraction vs vLLM, max |diff| vs vLLM in bf16 ulps of the vLLM
value, and max abs / rel error of both kernels vs the fp32 reference (so the new kernel can be judged
as 'no worse than vLLM', i.e. bf16 rounding noise)."""
import argparse
import sys

import torch

sys.path.insert(0, "/work/kernels/dense-fp8-gemm")
from common import SHAPES, load_weight, make_act, quant_per_token  # noqa: E402

from vllm import _custom_ops as ops  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--impl", default="skinny")  # skinny | dispatch | cx:<cfg>
ap.add_argument("--ms", default="1,2,3,5,7,8,9,12,15,16,17,24,31,33,48,63,64")
ap.add_argument("--layers", default="0,1,2,3,7,38,39")
args = ap.parse_args()


def get_impl(name):
    if name == "skinny":
        import skinny.build as sb
        mod = sb.load()
        from skinny.dispatch import pick
        def f(a, a_s, w, w_s):
            out = torch.empty(a.shape[0], w.shape[0], dtype=torch.bfloat16, device="cuda")
            kw, kb = pick(a.shape[0], w.shape[0], w.shape[1]) or (2, 4)
            mod.mm(out, a, w, a_s.reshape(-1), w_s, kw, kb)
            return out
        return f
    if name == "dispatch":
        from skinny.dispatch import dense_fp8_mm
        return lambda a, a_s, w, w_s: dense_fp8_mm(a, w.t(), a_s, w_s)
    if name.startswith("cx:"):
        import cutlass_ext.build as cb
        mod = cb.load()
        cfg = name[3:]
        def f(a, a_s, w, w_s):
            out = torch.empty(a.shape[0], w.shape[0], dtype=torch.bfloat16, device="cuda")
            mod.run(cfg, out, a, w.t(), a_s.reshape(-1).contiguous(), w_s, 0, 1, 1, 0)
            return out
        return f
    raise ValueError(name)


f = get_impl(args.impl)
ms = [int(x) for x in args.ms.split(",")]
layers = [int(x) for x in args.layers.split(",")]
worst = {"ulp": 0, "rel_new": 0.0, "rel_vllm": 0.0}
print(f"{'shape':9s} {'layer':>5s} {'M':>6s} {'bitEq%':>7s} {'max_ulp':>7s} {'maxabs_new':>10s} {'rel_new':>9s} "
      f"{'maxabs_vllm':>11s} {'rel_vllm':>9s}")
for sn in SHAPES:
    lys = [None] if sn == "lm_head" else [l for l in layers if (l % 4 == 3) == sn.startswith("attn")]
    for ly in lys:
        w, s = load_weight(sn, ly)
        for M in ms:
            x = make_act(M, w.shape[1], seed=1000 * M + (ly or 0))
            a, a_s = quant_per_token(x)
            base = ops.cutlass_scaled_mm(a, w.t(), a_s, s, torch.bfloat16)
            new = f(a, a_s, w, s)
            torch.cuda.synchronize()
            ref = (a.float() @ w.float().t()) * a_s.view(-1, 1) * s.view(1, -1)
            eq = (new.view(torch.int16) == base.view(torch.int16)).float().mean().item() * 100
            # difference in bf16 ulps (same sign region -> integer distance of the bit patterns)
            bi, ni = base.view(torch.int16).int(), new.view(torch.int16).int()
            same = (bi < 0) == (ni < 0)
            ulp = ((bi - ni).abs() * same).max().item()
            dn = (new.float() - ref).abs().max().item()
            dv = (base.float() - ref).abs().max().item()
            rm = ref.abs().max().item()
            worst["ulp"] = max(worst["ulp"], ulp)
            worst["rel_new"] = max(worst["rel_new"], dn / rm)
            worst["rel_vllm"] = max(worst["rel_vllm"], dv / rm)
            print(f"{sn:9s} {str(ly):>5s} {M:6d} {eq:7.2f} {ulp:7d} {dn:10.3g} {dn/rm:9.2e} {dv:11.3g} {dv/rm:9.2e}",
                  flush=True)
        del w
print("WORST", worst)
