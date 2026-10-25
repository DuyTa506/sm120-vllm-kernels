"""Correctness: glue-free path vs patch-01 path (bitwise) and vs an FP32 reference.

1. exhaustive BF16 -> E4M3 check of the in-kernel Q conversion (all 65536 BF16 bit
   patterns, q_scale 1.0 and a non-trivial q_scale) through a full FMHA launch is
   impractical, so it is checked on a Q tensor that contains every BF16 value.
2. varlen batches with cached prefixes, HND and NHD layouts, random page placement:
   output of new path == output of baseline path (max abs diff must be 0).
3. sanity vs an FP32 attention reference on the same FP8 operands.
"""

import sys
import time

import torch

from common import (D, HKV, HQ, PAGE, baseline_meta, fi_kernel, make_batch, make_pool,
                    make_query, new_meta, quant_q_baseline, ref_attention, run_baseline,
                    run_new)
from glue_fmha import get_kernel

torch.manual_seed(0)
dev = "cuda"
SM_SCALE = D ** -0.5
K_SCALE, V_SCALE = 0.0234375, 0.01953125  # realistic per-tensor kv scales (powers-of-2-ish)

fails = 0


def check(name, a, b, tol=0.0):
    global fails
    diff = (a.float() - b.float()).abs()
    nan_a, nan_b = torch.isnan(a.float()), torch.isnan(b.float())
    same_nan = bool((nan_a == nan_b).all())
    d = diff[~nan_a & ~nan_b]
    mx = d.max().item() if d.numel() else 0.0
    rel = (mx / max(b.float()[~nan_b].abs().max().item(), 1e-30)) if d.numel() else 0.0
    ok = same_nan and mx <= tol
    fails += not ok
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}: max_abs={mx:.3e} max_rel(to max|ref|)={rel:.3e}"
          f" nan_match={same_nan}")
    return mx


t0 = time.time()
fi = fi_kernel()
new = get_kernel(torch.bfloat16, torch.bfloat16, HQ, HKV, D, PAGE)
print(f"compiled kernels in {time.time() - t0:.1f}s")

# ---------------------------------------------------------------- 1. Q conversion
print("1. in-kernel BF16->E4M3 Q conversion (every BF16 bit pattern in Q)")
for q_scale in (1.0, 0.7071):
    kern = new if q_scale == 1.0 else get_kernel(torch.bfloat16, torch.bfloat16, HQ, HKV, D,
                                                 PAGE, q_scale=q_scale)
    # Q holds every finite BF16 bit pattern (NaN/inf -> 0); both paths must give a
    # bitwise identical attention output, which they only do if the FP8 Q matches.
    allbits = torch.arange(65536, dtype=torch.int32, device=dev).to(torch.int16).view(torch.bfloat16)
    q = allbits.reshape(16, HQ, D).clone()
    q = torch.where(torch.isfinite(q), q, torch.zeros_like(q))
    raw, k, v = make_pool(64)
    b = make_batch([(16, 200)], 64, seed=3)
    out_a = torch.empty(16, HQ, D, dtype=torch.bfloat16, device=dev)
    out_b = torch.empty_like(out_a)
    # tiny softmax scale keeps the softmax unsaturated so every Q element matters
    run_baseline(fi, q, k, v, out_a, baseline_meta(b), b["max_q"], 1e-3, q_scale, K_SCALE, V_SCALE)
    run_new(kern, q, k, v, out_b, new_meta(b), b["max_q"], 1e-3, q_scale, K_SCALE, V_SCALE)
    torch.cuda.synchronize()
    check(f"all-bf16 Q, q_scale={q_scale}: attention output new vs baseline", out_b, out_a)

# a direct check of the conversion formula itself vs torch on all BF16 values
x = torch.arange(65536, dtype=torch.int32, device=dev).to(torch.int16).view(torch.bfloat16)
for q_scale in (1.0, 0.7071, 3.0):
    ref = quant_q_baseline(x, q_scale).view(torch.uint8)
    inv = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(q_scale, dtype=torch.float32)
    emu = (x.float() * inv.item()).to(torch.bfloat16) if q_scale != 1.0 else x
    emu = emu.float().clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    nan = torch.isnan(quant_q_baseline(x, q_scale).float())
    bad = ((ref != emu) & ~nan).sum().item()
    print(f"  formula bf16(x*fp32(1/s)) then f32->e4m3 vs torch, q_scale={q_scale}: mismatches={bad}")
    fails += bad != 0

from q_quant import q_quant
for q_scale in (1.0, 0.7071, 3.0):
    ref = quant_q_baseline(x, q_scale)
    got = q_quant(x.contiguous(), q_scale)
    nan = torch.isnan(ref.float())
    bad = ((ref.view(torch.uint8) != got.view(torch.uint8)) & ~nan).sum().item()
    bad += (torch.isnan(got.float()) != nan).sum().item()
    print(f"  Triton fused q_quant vs torch, all bf16 values, q_scale={q_scale}: mismatches={bad}")
    fails += bad != 0

# ---------------------------------------------------------------- 2. varlen bitwise
print("2. varlen batches, new vs baseline (expect bitwise equal)")
cases = [
    ("single 8k", [(8192, 8192)]),
    ("single 8k chunk on 8k prefix", [(8192, 16384)]),
    ("varlen mix + prefixes", [(257, 257), (1000, 3000), (64, 64), (3333, 3333), (512, 4097)]),
    ("tiny + odd", [(300, 301), (1, 129), (777, 777), (2048, 2111)]),
]
for layout in ("HND", "NHD"):
    for name, seqs in cases:
        nb = sum(-(-kv // PAGE) for _, kv in seqs) + 37
        raw, k, v = make_pool(nb, layout=layout, seed=len(name))
        b = make_batch(seqs, nb, seed=len(name) + 1)
        q = make_query(b["total_q"], seed=len(name) + 2)
        out_a = torch.empty(b["total_q"], HQ, D, dtype=torch.bfloat16, device=dev)
        out_b = torch.full_like(out_a, float("nan"))
        run_baseline(fi, q, k, v, out_a, baseline_meta(b), b["max_q"], SM_SCALE, 1.0, K_SCALE, V_SCALE)
        run_new(new, q, k, v, out_b, new_meta(b), b["max_q"], SM_SCALE, 1.0, K_SCALE, V_SCALE)
        torch.cuda.synchronize()
        check(f"{layout} {name}", out_b, out_a)
        if b["total_q"] <= 9000 and layout == "HND":
            ref = ref_attention(q, k, v, b, SM_SCALE, 1.0, K_SCALE, V_SCALE)
            print("    vs fp32 reference:", end="")
            mx_new = (out_b.float() - ref).abs().max().item()
            mx_old = (out_a.float() - ref).abs().max().item()
            print(f" new max_abs={mx_new:.3e}  baseline max_abs={mx_old:.3e}  (|ref| max {ref.abs().max().item():.3f})")
            fails += mx_new > 2e-2

print("ALL OK" if fails == 0 else f"{fails} FAILURES")
sys.exit(1 if fails else 0)
