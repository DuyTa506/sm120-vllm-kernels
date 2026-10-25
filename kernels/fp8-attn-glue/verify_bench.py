"""Verifier's re-timing of patch-01 path (A) vs glue-free kernel (D).

Same work on both sides (same Q, pool, metadata; metadata built outside the timed region on
both sides, as both vLLM paths cache it per step). Interleaved round-robin, CUDA events,
eager launches (no CUDA graphs; vLLM runs prefill attention eagerly). Three cache regimes:
  flush : 256 MB L2 flush before each rep (author's method; Q and K/V cold)
  qhot  : L2 flush, then an untimed copy kernel writes Q (like RoPE/q-norm producing Q
          right before attention), so Q is partly L2-resident as in the model
  warm  : no flush (steady state, everything that fits stays in L2)
Also a sync-free "stream" mode: 20 back-to-back calls between one pair of events, so
CPU launch overhead of the extra glue launches is included the way it would be if the
CPU were not ahead of the GPU.
Reports median and IQR of per-call us; A-D and x11 calls per forward.
"""

import argparse
import statistics

import torch

from common import (D, HKV, HQ, PAGE, baseline_meta, fi_kernel, make_batch, make_pool,
                    make_query, new_meta, run_baseline, run_new)
from glue_fmha import get_kernel

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=60)
ap.add_argument("--cases", default="8k,varlen,2k,512,16k")
args = ap.parse_args()

CASES = {
    "8k": [(8192, 8192)],
    "16k": [(16384, 16384)],
    "2k": [(2048, 2048)],
    "512": [(512, 512)],
    "varlen": [(2048, 2048), (1024, 5120), (3000, 3000), (2120, 2120)],
}
SM, KS, VS = D ** -0.5, 0.0234375, 0.01953125
flush_buf = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
fi = fi_kernel()
new = get_kernel(torch.bfloat16, torch.bfloat16, HQ, HKV, D, PAGE)
NB = 4096
raw, k, v = make_pool(NB)


def q25_75(xs):
    xs = sorted(xs)
    return xs[len(xs) // 4], xs[(3 * len(xs)) // 4]


for cname in args.cases.split(","):
    seqs = CASES[cname]
    b = make_batch(seqs, NB, seed=7)
    q_src = make_query(b["total_q"])
    q = q_src.clone()
    out = torch.empty(b["total_q"], HQ, D, dtype=torch.bfloat16, device="cuda")
    bm, nm = baseline_meta(b), new_meta(b)
    fA = lambda: run_baseline(fi, q, k, v, out, bm, b["max_q"], SM, 1.0, KS, VS)  # noqa: E731
    fD = lambda: run_new(new, q, k, v, out, nm, b["max_q"], SM, 1.0, KS, VS)  # noqa: E731
    fns = {"A": fA, "D": fD}
    for f in fns.values():
        for _ in range(5):
            f()
    torch.cuda.synchronize()
    for mode in ("flush", "qhot", "warm"):
        ts = {n: [] for n in fns}
        names = list(fns)
        for i in range(args.reps):
            for j in range(len(names)):
                n = names[(i + j) % len(names)]
                if mode in ("flush", "qhot"):
                    flush_buf.zero_()
                if mode == "qhot":
                    q.copy_(q_src)
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                fns[n]()
                e.record()
                e.synchronize()
                ts[n].append(s.elapsed_time(e) * 1e3)
        mA, mD = statistics.median(ts["A"]), statistics.median(ts["D"])
        a25, a75 = q25_75(ts["A"])
        d25, d75 = q25_75(ts["D"])
        print(f"{cname:7s} {mode:6s} A {mA:9.1f} [{a25:8.1f},{a75:8.1f}]  D {mD:9.1f} "
              f"[{d25:8.1f},{d75:8.1f}]  A-D {mA - mD:7.1f} us  x11 = {11 * (mA - mD) / 1e3:.3f} ms")
    # back-to-back stream mode (no per-call sync; launch overhead visible if GPU starves)
    res = {}
    for rnd in range(3):
        for n, f in fns.items():
            torch.cuda.synchronize()
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(20):
                f()
            e.record()
            e.synchronize()
            res.setdefault(n, []).append(s.elapsed_time(e) * 1e3 / 20)
    mA, mD = statistics.median(res["A"]), statistics.median(res["D"])
    print(f"{cname:7s} stream A {mA:9.1f}  D {mD:9.1f}  A-D {mA - mD:7.1f} us (20 back-to-back, warm)")
