"""Time each piece of the patch-01 FP8 FMHA prefill glue and the glue-free replacement.

CUDA events around each call, L2 flushed (256 MB memset) before every timed rep so
the K/V pages and Q are cold as they are in the model (other layers run in between),
5 warmups, median of N reps. Eager launches (vLLM runs prefill attention outside CUDA
graphs in piecewise mode). Usage: bench.py [--reps 30] [--sweep] [--cases 8k,32k]
"""

import argparse
import statistics

import torch
from cutlass.cute.typing import Float32, Int32

from common import (D, HKV, HQ, LOG2E, PAGE, baseline_meta, causal_flops, fi_kernel,
                    make_batch, make_pool, make_query, new_meta, quant_q_baseline,
                    run_baseline, run_new)
from glue_fmha import get_kernel
from q_quant import q_quant

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=30)
ap.add_argument("--sweep", action="store_true", help="tile/scheduler sweep of the FMHA")
ap.add_argument("--cases", default="8k,16k,32k,32k-chunk,varlen")
args = ap.parse_args()

CASES = {
    "8k": [(8192, 8192)],
    "16k": [(16384, 16384)],
    "32k": [(32768, 32768)],
    "32k-chunk": [(16384, 32768)],  # 2nd chunk of a 32k prompt at max-num-batched-tokens 16384
    "varlen": [(2048, 2048), (1024, 5120), (3000, 3000), (2120, 2120)],
}
SM, KS, VS = D ** -0.5, 0.0234375, 0.01953125
flush_buf = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")


def bench(fn, reps=args.reps, flush=True):
    for _ in range(5):
        fn()
    ts = []
    for _ in range(reps):
        if flush:
            flush_buf.zero_()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    return statistics.median(ts)


def bench_interleaved(fns, reps=None):
    reps = reps or max(args.reps, 40)
    for f in fns.values():
        for _ in range(3):
            f()
    ts = {k: [] for k in fns}
    names = list(fns)
    for i in range(reps):
        for j in range(len(names)):
            name = names[(i + j) % len(names)]
            flush_buf.zero_()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fns[name]()
            e.record()
            e.synchronize()
            ts[name].append(s.elapsed_time(e) * 1e3)
    return {k: statistics.median(v) for k, v in ts.items()}


fi = fi_kernel()
new = get_kernel(torch.bfloat16, torch.bfloat16, HQ, HKV, D, PAGE)
new_fp8q = get_kernel(torch.float8_e4m3fn, torch.bfloat16, HQ, HKV, D, PAGE)

# realistic pool size: production uses ~0.78 GPU mem; any pool >> L2 behaves the same
NB = 4096
raw, k, v = make_pool(NB)

print(f"{'case':10s} {'piece':46s} {'us':>9s}  notes")
summary = {}
for cname in args.cases.split(","):
    seqs = CASES[cname]
    b = make_batch(seqs, NB, seed=7)
    q = make_query(b["total_q"])
    out = torch.empty(b["total_q"], HQ, D, dtype=torch.bfloat16, device="cuda")
    bm, nm = baseline_meta(b), new_meta(b)
    idx, bt, seqlens, cu_q = bm
    kc, vc = k.index_select(0, idx), v.index_select(0, idx)
    q8 = quant_q_baseline(q, 1.0)
    fl = causal_flops(seqs)
    mq = Int32(b["max_q"])
    r = {}
    r["K index_select"] = bench(lambda: k.index_select(0, idx))
    r["V index_select"] = bench(lambda: v.index_select(0, idx))
    r["Q clamp+cast (q_scale=1)"] = bench(lambda: quant_q_baseline(q, 1.0))
    r["Q div+clamp+cast (q_scale!=1)"] = bench(lambda: quant_q_baseline(q, 0.5))
    r["Q fused Triton quant (alt.)"] = bench(lambda: q_quant(q, 1.0))
    # The FMHA-sized variants are timed interleaved (round-robin each rep) because GPU
    # clocks drift with temperature/power over a run; sequential blocks were +-5-10%.
    fns = {
        "FMHA only (FlashInfer, pre-gathered KV, fp8 Q)": lambda: fi(
            q8, kc, vc, out, None, Float32(SM * KS * LOG2E), Float32(VS), seqlens, cu_q, bt,
            None, mq, False),
        "FMHA only (strided KV, fp8 Q pre-quantised)": lambda: new_fp8q(
            q8, k, v, out, None, Float32(SM * KS * LOG2E), Float32(VS),
            nm[1], nm[2], nm[0], None, mq, False),
        "A baseline: gather+torch quant+FMHA": lambda: run_baseline(
            fi, q, k, v, out, bm, b["max_q"], SM, 1.0, KS, VS),
        "B strided KV + torch quant": lambda: new_fp8q(
            quant_q_baseline(q, 1.0), k, v, out, None, Float32(SM * KS * LOG2E), Float32(VS),
            nm[1], nm[2], nm[0], None, mq, False),
        "C strided KV + Triton quant": lambda: new_fp8q(
            q_quant(q, 1.0), k, v, out, None, Float32(SM * KS * LOG2E), Float32(VS),
            nm[1], nm[2], nm[0], None, mq, False),
        "D new: strided KV + bf16 Q in kernel": lambda: run_new(
            new, q, k, v, out, nm, b["max_q"], SM, 1.0, KS, VS),
    }
    r.update(bench_interleaved(fns))
    for kname, us in r.items():
        note = ""
        if kname[:2] in ("A ", "B ", "C ", "D ") or "FMHA" in kname:
            note = f"{fl / us / 1e6:7.1f} TFLOPS"
        print(f"{cname:10s} {kname:46s} {us:9.1f}  {note}")
    saved = r["A baseline: gather+torch quant+FMHA"] - r["D new: strided KV + bf16 Q in kernel"]
    print(f"{cname:10s} {'=> saved per layer (A - D)':46s} {saved:9.1f}  x11 calls/forward (10 layers + MTP) = {saved * 11 / 1e3:.3f} ms")
    summary[cname] = r

if args.sweep:
    print("\nFMHA tile/scheduler sweep (glue-free kernel, bf16 Q, strided KV)")
    for cname in args.cases.split(","):
        seqs = CASES[cname]
        b = make_batch(seqs, NB, seed=7)
        q = make_query(b["total_q"])
        out = torch.empty(b["total_q"], HQ, D, dtype=torch.bfloat16, device="cuda")
        nm = new_meta(b)
        fl = causal_flops(seqs)
        for qt in (128, 64):
            for kt in (128, 64):
                for bal in (True, False):
                    try:
                        fn = get_kernel(torch.bfloat16, torch.bfloat16, HQ, HKV, D, PAGE,
                                        kv_tile=kt, q_tile=qt, balanced=bal)
                    except Exception as ex:  # noqa: BLE001
                        print(f"{cname:10s} q{qt} kv{kt} bal{int(bal)}: compile failed {ex}")
                        continue
                    us = bench(lambda: run_new(fn, q, k, v, out, nm, b["max_q"], SM, 1.0, KS, VS))
                    print(f"{cname:10s} q_tile={qt:3d} kv_tile={kt:3d} balanced={int(bal)} "
                          f"{us:9.1f} us {fl / us / 1e6:7.1f} TFLOPS")
