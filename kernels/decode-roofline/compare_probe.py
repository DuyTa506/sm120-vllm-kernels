"""Compare two greedy quality-probe runs (eval/quality_probe.py output): first divergence position per
prompt, identical-prefix logprob deltas.  usage: python3 -I compare_probe.py a.json b.json"""
import json, sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
tot_same = tot = 0
maxd = 0.0
for x, y in zip(a, b):
    n = min(len(x["tokens"]), len(y["tokens"]))
    div = next((i for i in range(n) if x["tokens"][i] != y["tokens"][i]), n)
    d = max((abs(x["lp"][i] - y["lp"][i]) for i in range(div) if x["lp"][i] is not None and y["lp"][i] is not None), default=0.0)
    maxd = max(maxd, d)
    tot_same += div; tot += n
    print(f"  {'identical' if div == n else f'diverges at token {div:3d}'} / {n}   max |dlogprob| before divergence {d:.4f}   {x['prompt'][:50]}")
print(f"identical-prefix tokens {tot_same}/{tot}; max |dlogprob| on shared prefix {maxd:.4f}")
