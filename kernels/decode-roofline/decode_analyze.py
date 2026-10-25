"""Per-step breakdown of a vLLM decode nsys capture (graph-level or --cuda-graph-trace=node).

usage: python3 -I decode_analyze.py <trace.sqlite> [--seq N] [--top N]

GPU work is grouped into "launch units" by CUPTI correlationId: one cudaGraphLaunch (all its kernels
share the launch's correlationId) or one eager kernel / memcpy / memset.
A decode step starts at each launch of the target-model graph (the graph with the largest median
duration among frequently launched graphs). For each step we report span, GPU-busy union, time in
the target graph, time in other graphs (MTP drafter pieces), eager kernels, and idle gaps.
"""
import collections, sqlite3, statistics, sys

args = sys.argv[1:]
path = args[0]
SEQ = int(args[args.index("--seq") + 1]) if "--seq" in args else 0
TOP = int(args[args.index("--top") + 1]) if "--top" in args else 45
db = sqlite3.connect(path)
S = dict(db.execute("SELECT id, value FROM StringIds"))

ops = []  # (start, end, corr, kind, name, is_graph_node)
for s, e, c, n, g in db.execute(
        "SELECT start, end, correlationId, demangledName, graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL"):
    ops.append((s, e, c, "k", S.get(n, "?"), g is not None))
for tbl, kind in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
    try:
        for s, e, c, g in db.execute(f"SELECT start, end, correlationId, graphNodeId FROM {tbl}"):
            ops.append((s, e, c, kind, kind, g is not None))
    except sqlite3.OperationalError:
        for s, e, c in db.execute(f"SELECT start, end, correlationId FROM {tbl}"):
            ops.append((s, e, c, kind, kind, False))
api = {c: S.get(n, "?") for c, n in db.execute("SELECT correlationId, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
api_t = {c: (s, e) for c, s, e in db.execute("SELECT correlationId, start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME")}

# graph-level traces: one row per graph launch
graph_rows = []
try:
    cols = [r[1] for r in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_GRAPH_TRACE)")]
    if cols:
        gid = "graphId" if "graphId" in cols else "NULL"
        graph_rows = db.execute(f"SELECT start, end, correlationId, {gid} FROM CUPTI_ACTIVITY_KIND_GRAPH_TRACE").fetchall()
except sqlite3.OperationalError:
    pass

units = collections.defaultdict(lambda: {"s": 1 << 62, "e": 0, "kern": [], "graph": False})
for s, e, c, kind, name, gnode in ops:
    u = units[("o", c)]
    u["s"] = min(u["s"], s); u["e"] = max(u["e"], e)
    u["kern"].append((s, e, name, kind))
    if gnode or "Graph" in api.get(c, ""):
        u["graph"] = True
for s, e, c, g in graph_rows:
    u = units[("o", c)]
    u["s"] = min(u["s"], s); u["e"] = max(u["e"], e); u["graph"] = True; u["gid"] = g
U = sorted(units.values(), key=lambda u: u["s"])
for u in U:
    u["d"] = u["e"] - u["s"]
    u["sig"] = (u.get("gid"), len(u["kern"])) if u["graph"] else None

# target graph: among graph units, group by signature; pick the one whose launches are frequent and longest
gstat = collections.defaultdict(list)
for u in U:
    if u["graph"]:
        gstat[u["sig"]].append(u["d"])
if not gstat:
    sys.exit("no graph launches found")
cand = [(statistics.median(v), k, len(v)) for k, v in gstat.items() if len(v) >= max(3, 0.2 * max(map(len, gstat.values())))]
cand.sort(reverse=True)
print("graph signatures (gid, nkernels): count, median us")
for k, v in sorted(gstat.items(), key=lambda x: -len(x[1]))[:12]:
    print(f"  {str(k):24s} x{len(v):5d}  med {statistics.median(v)/1e3:8.1f} us  max {max(v)/1e3:8.1f}")


def union(iv):
    tot, cs, ce = 0, None, None
    for s, e in sorted(iv):
        if ce is None or s > ce:
            if ce is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + (ce - cs if ce is not None else 0)


# step boundaries: launches of the largest-duration graph signatures that look like the target
# forward. With MTP there are often several sizes (padded token counts); take every graph with
# >= 70% of the max kernel count of the dominant signature as a "target forward".
main_sig = cand[0][1]
main_n = main_sig[1]
is_main = lambda u: u["graph"] and (len(u["kern"]) >= 0.7 * main_n if main_n > 1 else u["sig"] == main_sig)
mains = [i for i, u in enumerate(U) if is_main(u)]
steps = []
for a, b in zip(mains, mains[1:]):
    seg = U[a:b]
    t0, t1 = U[a]["s"], U[b]["s"]
    if t1 - t0 > 200e6:  # >200 ms: idle between requests, not a step
        continue
    busy = union([(k[0], k[1]) for u in seg for k in u["kern"]] + [(u["s"], u["e"]) for u in seg if not u["kern"]])
    tg = U[a]["d"]
    other_graph = sum(u["d"] for u in seg[1:] if u["graph"])
    eager = [u for u in seg if not u["graph"]]
    eager_t = sum(k[1] - k[0] for u in eager for k in u["kern"])
    n_eager = sum(len(u["kern"]) for u in eager)
    gaps = []
    le = seg[0]["e"]
    for u in seg[1:]:
        if u["s"] > le:
            gaps.append(u["s"] - le)
        le = max(le, u["e"])
    gaps.append(t1 - le if t1 > le else 0)
    steps.append(dict(span=t1 - t0, busy=busy, tg=tg, og=other_graph, eager=eager_t, neager=n_eager,
                      ngraph=sum(1 for u in seg if u["graph"]), maxgap=max(gaps), idx=a, ntok=len(U[a]["kern"])))

if not steps:
    sys.exit("no steps")
med = lambda key: statistics.median(s[key] for s in steps)
print(f"\n{len(steps)} decode steps (target graph = sig {main_sig})")
print(f"  median step span        {med('span')/1e3:8.1f} us")
print(f"  median GPU busy (union) {med('busy')/1e3:8.1f} us  ({100*med('busy')/med('span'):.0f}% of span)")
print(f"  median target graph     {med('tg')/1e3:8.1f} us")
print(f"  median other graphs     {med('og')/1e3:8.1f} us  ({med('ngraph')-1:.0f} launches)")
print(f"  median eager kernels    {med('eager')/1e3:8.1f} us  ({med('neager'):.0f} kernels/memops)")
print(f"  median idle             {(med('span')-med('busy'))/1e3:8.1f} us; median largest single gap {med('maxgap')/1e3:.1f} us")
spans = sorted(s["span"] for s in steps)
print(f"  span p10/p50/p90        {spans[len(spans)//10]/1e3:.0f} / {spans[len(spans)//2]/1e3:.0f} / {spans[9*len(spans)//10]/1e3:.0f} us")

# timeline of one representative step
st = sorted(steps, key=lambda s: abs(s["span"] - med("span")))[0] if not SEQ else steps[SEQ]
a = st["idx"]
b = next((i for i in mains if i > a), len(U))
print(f"\n== timeline of a median step ({st['span']/1e3:.0f} us); offsets in us from step start")
t0 = U[a]["s"]; le = t0
for u in U[a:b]:
    gap = u["s"] - le
    if u["graph"]:
        desc = f"GRAPH {len(u['kern'])} kernels"
    else:
        desc = (u["kern"][0][2][:90] if u["kern"] else "?")
    print(f"  {(u['s']-t0)/1e3:8.1f} +{u['d']/1e3:7.1f}  gap {gap/1e3:6.1f}  {desc}")
    le = max(le, u["e"])

# per-kernel table (only meaningful with node-level traces), per step, split by phase
by = collections.defaultdict(lambda: [0, 0])
nsteps = len(steps)
for s in steps:
    a = s["idx"]; b = next((i for i in mains if i > a), len(U))
    for j, u in enumerate(U[a:b]):
        ph = "target" if j == 0 else ("graph2" if u["graph"] else "eager")
        for k in u["kern"]:
            key = (ph, k[2])
            by[key][0] += k[1] - k[0]; by[key][1] += 1
ph_tot = collections.Counter()
for (ph, n), (t, c) in by.items():
    ph_tot[ph] += t
print("\n== kernel time per step by phase (us): " + ", ".join(f"{p} {t/nsteps/1e3:.1f}" for p, t in ph_tot.items()))
print(f"== top {TOP} kernels (us per step, launches per step, avg us)")
for (ph, n), (t, c) in sorted(by.items(), key=lambda x: -x[1][0])[:TOP]:
    print(f"  {t/nsteps/1e3:8.1f} x{c/nsteps:5.1f} avg {t/c/1e3:7.1f}  [{ph:6s}] {n[:120]}")
