"""Phase analysis of an idle_bench result: later turns that start after the cold first wave has finished
(steady state) vs during it. usage: analyze.py <result.json> [...]"""
import json, sys
from common import pct

for f in sys.argv[1:]:
    d = json.load(open(f)); T = d["turns"]; s = d["summary"]
    first_end = max((r["t_start"] + r["ttft"]) for r in T if r["turn"] == 0) if all(
        any(r["sid"] == i and r["turn"] == 0 for r in T) for i in range(s["sessions"])) else None
    print(f"== {f}: wall {s['wall_s']:.0f}s, turns {len(T)} (+{s.get('censored_inflight',0)} censored), "
          f"first wave done at {first_end and round(first_end)}s")
    later = [r for r in T if r["turn"] > 0]
    cens = [c for c in d.get("censored", []) if c["turn"] > 0 and c["ttft"] is None]
    for name, sel in [("all later", lambda r: True),
                      ("later, start before first wave done", lambda r: first_end is None or r["t_start"] < first_end),
                      ("later, start after first wave done", lambda r: first_end is not None and r["t_start"] >= first_end)]:
        rs = [r for r in later if sel(r)]
        cs = [c["waited"] for c in cens if sel(c)]
        tt = [r["ttft"] for r in rs] + cs
        if not tt:
            continue
        rc = [r["recompute"] for r in rs]
        print(f"  {name:38s} n={len(rs):3d}+{len(cs)}c  TTFT p50 {1000*pct(tt,.5):8.0f} p90 {1000*pct(tt,.9):8.0f} ms"
              f"  recompute mean {sum(rc)/max(len(rc),1):7.0f} p50 {pct(rc,.5)} p90 {pct(rc,.9)}"
              f"  new-tokens mean {sum(r['new_tokens'] for r in rs)/max(len(rs),1):.0f}  >8k: {sum(x>8192 for x in rc)}")

# later-turn TTFT and recompute per 120 s bin of the turn's start time
for f in sys.argv[1:]:
    d = json.load(open(f))
    rows = [(r["t_start"], r["ttft"], r["recompute"]) for r in d["turns"] if r["turn"] > 0]
    rows += [(c["t_start"], c["waited"], None) for c in d.get("censored", []) if c["turn"] > 0 and c["ttft"] is None]
    print(f"-- {f}: later turns by start time (TTFT p50 s, mean recompute tokens, n; censored count as waited)")
    for b in range(0, 800, 120):
        sel = [x for x in rows if b <= x[0] < b + 120]
        if sel:
            rc = [x[2] for x in sel if x[2] is not None]
            print(f"   {b:4d}-{b+120:4d}s  n={len(sel):3d}  TTFT p50 {pct([x[1] for x in sel], .5):7.1f}  "
                  f"recompute {sum(rc)/max(len(rc),1):8.0f}")
