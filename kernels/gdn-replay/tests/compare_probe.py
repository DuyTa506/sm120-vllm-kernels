"""usage: compare_probe.py a.json b.json  -> per prompt: identical?, first divergence, max |dlogprob| on shared prefix"""
import json, sys
a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
same = 0; tot = 0; shared = 0; total_tok = 0
for i, (x, y) in enumerate(zip(a, b)):
    n = min(len(x["tokens"]), len(y["tokens"]))
    d = next((k for k in range(n) if x["tokens"][k] != y["tokens"][k]), None)
    full = d is None and len(x["tokens"]) == len(y["tokens"])
    same += full; tot += 1
    shared += (n if d is None else d); total_tok += max(len(x["tokens"]), len(y["tokens"]))
    m = max([abs(p - q) for p, q in zip(x["lp"][: (d if d is not None else n)], y["lp"][: (d if d is not None else n)]) if p is not None and q is not None] or [0])
    print(f"prompt {i:2d}: {'IDENTICAL' if full else 'diverge@%s' % d}  len {len(x['tokens'])}/{len(y['tokens'])}  max|dlp| on shared prefix {m:.3g}")
print(f"SUMMARY identical prompts {same}/{tot}; shared-prefix tokens {shared}/{total_tok}")
