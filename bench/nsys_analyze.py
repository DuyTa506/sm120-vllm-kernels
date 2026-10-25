import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
ks = db.execute("SELECT start,end,'k' FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
gs = db.execute("SELECT start,end,'g' FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE").fetchall()
ev = sorted(ks + gs)
# a request = events separated from the next by > 3 ms of GPU idle
reqs, cur, le = [], [ev[0]], ev[0][1]
for r in ev[1:]:
    if r[0] - le > 3_000_000: reqs.append(cur); cur = [r]
    else: cur.append(r)
    le = max(le, r[1])
reqs.append(cur)
def union(rs):
    b, cs, ce = 0, None, None
    for s, e, _ in sorted(rs):
        if ce is None or s > ce:
            if ce is not None: b += ce - cs
            cs, ce = s, e
        else: ce = max(ce, e)
    return b + (ce - cs)
big = [r for r in reqs if len(r) >= 100]
print(f"{len(ks)} standalone kernels, {len(gs)} graph launches; {len(big)} busy windows")
tot_span = tot_busy = 0
for i, r in enumerate(big):
    span = max(x[1] for x in r) - r[0][0]; b = union(r)
    tot_span += span; tot_busy += b
    print(f"  window {i}: {sum(1 for x in r if x[2]=='k'):4d} kernels + {sum(1 for x in r if x[2]=='g'):3d} graphs  span {span/1e6:6.1f} ms  GPU busy {b/1e6:6.1f} ms = {100*b/span:3.0f}%")
print(f"  overall: span {tot_span/1e6:.1f} ms, busy {100*tot_busy/tot_span:.0f}%")
