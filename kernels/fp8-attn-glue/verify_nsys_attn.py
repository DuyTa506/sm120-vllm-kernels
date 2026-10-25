"""Verifier: compare two nsys sqlite captures (NOGLUE=0 vs 1) of single 8k prefills.

usage: python3 -I verify_nsys_attn.py ng0.sqlite ng1.sqlite
Per request window (GPU events separated by > 3 ms idle, as bench/nsys_analyze.py):
GPU span, and per-kernel-name total time per request; prints the names whose
per-request time differs most between the two captures (that is where patch 03 acts).
"""

import collections
import sqlite3
import statistics
import sys


def load(path):
    db = sqlite3.connect(path)
    names = dict(db.execute("SELECT id, value FROM StringIds"))
    ks = db.execute("SELECT start, end, demangledName, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    try:
        gs = db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE").fetchall()
    except sqlite3.OperationalError:
        gs = []
    ev = sorted([(s, e, names.get(d) or names.get(sn) or "?") for s, e, d, sn in ks]
                + [(s, e, "<cuda graph>") for s, e in gs])
    reqs, cur, le = [], [ev[0]], ev[0][1]
    for r in ev[1:]:
        if r[0] - le > 3_000_000:
            reqs.append(cur)
            cur = [r]
        else:
            cur.append(r)
        le = max(le, r[1])
    reqs.append(cur)
    big = [r for r in reqs if len(r) >= 100]
    spans = [(max(x[1] for x in r) - r[0][0]) / 1e6 for r in big]
    per_req = []
    for r in big:
        acc = collections.Counter()
        cnt = collections.Counter()
        for s, e, n in r:
            key = n[:110]
            acc[key] += (e - s) / 1e3
            cnt[key] += 1
        per_req.append((acc, cnt))
    keys = set().union(*[set(a) for a, _ in per_req])
    per_name = {k: [(a.get(k, 0.0), c.get(k, 0)) for a, c in per_req] for k in keys}
    return spans, per_name, len(big)


a_sp, a_pn, na = load(sys.argv[1])
b_sp, b_pn, nb = load(sys.argv[2])
print(f"A (NOGLUE=0): {na} requests, span ms: {[round(x, 2) for x in a_sp]}")
print(f"B (NOGLUE=1): {nb} requests, span ms: {[round(x, 2) for x in b_sp]}")
print(f"median span A {statistics.median(a_sp):.2f} ms, B {statistics.median(b_sp):.2f} ms, "
      f"diff {statistics.median(a_sp) - statistics.median(b_sp):.2f} ms; "
      f"min A {min(a_sp):.2f} B {min(b_sp):.2f}")


def med(lst, i):
    return statistics.median([x[i] for x in lst]) if lst else 0.0


rows = []
for key in set(a_pn) | set(b_pn):
    ta, tb = med(a_pn.get(key, []), 0), med(b_pn.get(key, []), 0)
    ca, cb = med(a_pn.get(key, []), 1), med(b_pn.get(key, []), 1)
    rows.append((ta - tb, ta, tb, ca, cb, key))
rows.sort(key=lambda r: -abs(r[0]))
print(f"\n{'dA-B us':>9s} {'A us':>9s} {'B us':>9s} {'nA':>5s} {'nB':>5s}  kernel (per request, median)")
for d, ta, tb, ca, cb, key in rows[:25]:
    print(f"{d:9.1f} {ta:9.1f} {tb:9.1f} {ca:5.0f} {cb:5.0f}  {key}")
tot_a = sum(r[1] for r in rows)
tot_b = sum(r[2] for r in rows)
print(f"\nsum of per-name medians: A {tot_a / 1e3:.2f} ms  B {tot_b / 1e3:.2f} ms  diff {(tot_a - tot_b) / 1e3:.3f} ms")
