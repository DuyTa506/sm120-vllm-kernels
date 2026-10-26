"""Paired per-turn TTFT comparison of two replay_subset.py outputs (same turns), split by context length."""
import json, statistics, sys
a, b = (json.load(open(f)) for f in sys.argv[1:3])
for c in [k for k in a if k != "args"]:
    ra = {(r["task"], r["turn"]): r for r in a[c]["turns"]}; rb = {(r["task"], r["turn"]): r for r in b[c]["turns"]}
    keys = [k for k in ra if k in rb and k[1] > 0]
    print(f"conc {c}: {len(keys)} paired later turns")
    for lo, hi in ((0, 30000), (30000, 70000), (70000, 200000)):
        ks = [k for k in keys if lo <= ra[k]["prompt"] < hi]
        if not ks: continue
        d = [1000 * (rb[k]["ttft"] - ra[k]["ttft"]) for k in ks]
        print(f"  ctx {lo//1000:3d}-{hi//1000:3d}k n={len(ks):3d}  base mean {1000*statistics.mean(ra[k]['ttft'] for k in ks):6.0f}"
              f"  new mean {1000*statistics.mean(rb[k]['ttft'] for k in ks):6.0f}  paired diff median {statistics.median(d):+6.0f} ms"
              f"  new faster in {sum(x < 0 for x in d)}/{len(d)}")
