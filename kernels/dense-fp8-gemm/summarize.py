"""Summarize a bench_sweep.py log: per (shape, M) the vLLM time and the best alternatives."""
import collections
import sys

rows = collections.defaultdict(list)
for line in open(sys.argv[1]):
    p = line.split()
    if len(p) >= 12 and p[1].isdigit():
        try:
            us = float(p[3])
        except ValueError:
            continue
        rows[(p[0], int(p[1]))].append(dict(us=us, impl=p[2], gbs=float(p[6]), eq=float(p[8]),
                                            maxd=float(p[9]), rel=p[11]))
top = int(sys.argv[2]) if len(sys.argv) > 2 else 3
filt = sys.argv[3] if len(sys.argv) > 3 else ""
for (sn, M), v in rows.items():
    base = next(x for x in v if x["impl"] == "vllm")
    v = sorted((x for x in v if filt in x["impl"] or x["impl"] == "vllm"), key=lambda x: x["us"])
    alts = "; ".join(f"{x['impl']} {x['us']:.1f} ({x['gbs']:.0f}GB/s, eq{x['eq']:.1f}%, maxd {x['maxd']:.3g})"
                     for x in v[:top])
    print(f"{sn:9s} M={M:5d} vllm {base['us']:7.1f}us {base['gbs']:5.0f}GB/s | {alts}")
