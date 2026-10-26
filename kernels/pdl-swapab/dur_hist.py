"""Kernel duration histogram and per-stream view of the target verify graph (node-level nsys trace).
usage: python3 -I dur_hist.py <node_trace.sqlite>
"""
import collections, sqlite3, statistics, sys
db = sqlite3.connect(sys.argv[1])
S = dict(db.execute("SELECT id, value FROM StringIds"))
by = collections.defaultdict(list)
for s, e, c, n, st, gx, gy, gz in db.execute("SELECT start,end,correlationId,demangledName,streamId,gridX,gridY,gridZ FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE graphNodeId IS NOT NULL"):
    by[c].append((s, e, S[n], st, gx * gy * gz))
for tbl, lab in (("CUPTI_ACTIVITY_KIND_MEMSET", "memset"), ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy")):  # graph memset/memcpy nodes
    for s, e, c, st in db.execute(f"SELECT start,end,correlationId,streamId FROM {tbl} WHERE graphNodeId IS NOT NULL"):
        by[c].append((s, e, lab, st, 0))
sizes = collections.Counter(len(v) for v in by.values())
tn = max(n for n, cnt in sizes.items() if cnt >= 20)
L = [sorted(v) for v in by.values() if len(v) == tn]
print(f"target graph {tn} kernels, {len(L)} launches")
# duration histogram (median over launches of count & total time per bin)
bins = [(0, 1500), (1500, 2500), (2500, 4000), (4000, 8000), (8000, 20000), (20000, 10**9)]
for lo, hi in bins:
    cnt = [sum(1 for k in ks if lo <= k[1] - k[0] < hi) for ks in L]
    tot = [sum(k[1] - k[0] for k in ks if lo <= k[1] - k[0] < hi) for ks in L]
    print(f"  dur [{lo/1e3:4.1f},{min(hi,99e3)/1e3:5.1f}) us: {statistics.median(cnt):6.0f} kernels, {statistics.median(tot)/1e3:8.1f} us")
# union / idle
idle, span, union_, s2 = [], [], [], collections.Counter()
for ks in L:
    t0 = ks[0][0]; t1 = max(k[1] for k in ks)
    u = 0; ce = None
    for k in ks:
        if ce is None or k[0] > ce:
            if ce is not None: u += ce - cs
            cs, ce = k[0], k[1]
        else:
            ce = max(ce, k[1])
    u += ce - cs
    span.append(t1 - t0); union_.append(u); idle.append(t1 - t0 - u)
    s2.update(k[3] for k in ks)
print(f"  median span {statistics.median(span)/1e3:.1f} us, union busy {statistics.median(union_)/1e3:.1f} us, idle {statistics.median(idle)/1e3:.1f} us")
print("  kernels per stream (all launches):", dict(s2))
# per-stream gaps (same stream consecutive)
ps = []
for ks in L:
    by_s = collections.defaultdict(list)
    for k in ks: by_s[k[3]].append(k)
    g = 0
    for v in by_s.values():
        v.sort()
        for a, b in zip(v, v[1:]):
            g += max(0, b[0] - a[1])
    ps.append(g)
print(f"  same-stream consecutive gaps total {statistics.median(ps)/1e3:.1f} us")
# gaps by size class of the second kernel: tiny (<3us) vs others
tiny = []
for ks in L:
    run_end = ks[0][1]; g = 0
    for a, b in zip(ks, ks[1:]):
        gap = max(0, b[0] - run_end)
        if b[1] - b[0] < 3000: g += gap
        run_end = max(run_end, b[1])
    tiny.append(g)
print(f"  idle gap directly before a <3us kernel: {statistics.median(tiny)/1e3:.1f} us")
# smallest-kernel list: top short kernel names by count
cn = collections.Counter(); ct = collections.Counter()
for k in L[len(L)//2]:
    if k[1]-k[0] < 3000:
        cn[k[2][:70]] += 1; ct[k[2][:70]] += k[1]-k[0]
print("  <3us kernels in one launch, by name (count, total us):")
for n, c in cn.most_common(14): print(f"    {c:4d} {ct[n]/1e3:6.1f} us  {n}")
