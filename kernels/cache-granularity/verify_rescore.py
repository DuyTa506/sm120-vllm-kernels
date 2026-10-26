"""Rescore verify_correctness.py output: a run counts as correct if it contains the code minus its last digit
(the P16 run generated only 14 tokens, which cuts the last digit). Also classifies outputs: correct / empty / wrong
(a different code, e.g. the code of the PREVIOUS prompt = stale state)."""
import json, re, sys
for f in sys.argv[1:]:
    d = json.load(open(f)); rows = d["rows"]
    def cls(t, e, stale=None):
        t = (t or "").replace(" ", "")
        if e[:-1] in t: return "ok"
        if stale and stale[:-1] in t: return "STALE"
        if not re.search(r"\d", t): return "empty"
        return "wrong"
    agg = {}
    for r in rows:
        if "expect" not in r: r["expect"] = None
        for k in ("cold", "hot"):
            if f"{k}_text" in r and r["expect"]:
                r[f"{k}_cls"] = cls(r[f"{k}_text"], r["expect"], r.get("stale"))
    print(f"== {f}")
    for r in rows:
        if r.get("expect"):
            print(f"  {r['case']:9s} len {r['len']:6d} cached {r.get('hot_cached')!s:6s} cold {r.get('cold_cls')} hot {r.get('hot_cls')}  tv {r.get('tv')}  agree {r.get('agree')}  cold='{r.get('cold_text','')[:18]}' hot='{r.get('hot_text','')[:18]}'")
    both = [r for r in rows if r.get("cold_cls") and r.get("hot_cls")]
    for k in ("cold", "hot"):
        print(f"  {k}: " + ", ".join(f"{c}={sum(r[k+'_cls']==c for r in both)}" for c in ("ok", "empty", "wrong", "STALE")))
    print("  wrong hot outputs:", [(r['case'], r['hot_text']) for r in both if r['hot_cls'] == 'wrong'])
