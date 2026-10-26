"""Verifier follow-up: is the outlier "exact retry restored from CPU" (A resent after reset) an offload effect or
just the hit-path numerics? For given (task, turn) prompts A, repeat N times each:
  cold      A(fresh salt)
  gpu-retry A(s); A(s)            (second one hits the GPU prefix cache)
  cpu-retry A(s); reset; A(s)     (second one restores from CPU)
and compare first-token top-20 distributions / greedy 32-token texts across groups.
usage: verify_repeat.py <out.json> [--cases 6:22,5:11,5:34] [--n 4]
"""
import argparse, json, math, time

from common import LONG_TRAJ, complete, post

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--cases", default="6:22,5:11,5:34")
p.add_argument("--n", type=int, default=4)
a = p.parse_args()
trajs = json.load(open(LONG_TRAJ))["trajectories"]


def reset():
    time.sleep(1.0)
    for _ in range(100):
        if post("/reset_prefix_cache").get("success"):
            return True
        time.sleep(0.2)
    return False


def tv(x, y):
    keys = set(x) | set(y)
    return 0.5 * sum(abs(math.exp(x.get(k, -30)) - math.exp(y.get(k, -30))) for k in keys)


out = []
for cs in a.cases.split(","):
    task, turn = map(int, cs.split(":"))
    A = trajs[task]["turns"][turn]["prompt_ids"]
    g = {"cold": [], "gpu": [], "cpu": []}
    for i in range(a.n):
        g["cold"].append(complete(A, f"rp-{cs}-c{i}", 32, logprobs=20))
        complete(A, f"rp-{cs}-g{i}", 1)
        g["gpu"].append(complete(A, f"rp-{cs}-g{i}", 32, logprobs=20))
        complete(A, f"rp-{cs}-p{i}", 1)
        reset()
        g["cpu"].append(complete(A, f"rp-{cs}-p{i}", 32, logprobs=20))
    rec = {"case": cs, "len": len(A), "cached": {k: [r["cached"] for r in v] for k, v in g.items()},
           "texts": {k: [r["text"] for r in v] for k, v in g.items()},
           "top1": {k: [max(r["top_logprobs"][0], key=r["top_logprobs"][0].get) for r in v] for k, v in g.items()}}
    lp = {k: [r["top_logprobs"][0] for r in v] for k, v in g.items()}
    pair = {}
    for x in g:
        for y in g:
            if x <= y:
                vals = [tv(u, w) for i, u in enumerate(lp[x]) for j, w in enumerate(lp[y]) if x != y or i < j]
                pair[f"{x}-{y}"] = sum(vals) / len(vals)
    rec["mean_pair_tv"] = pair
    rec["distinct_texts"] = {k: len(set(v)) for k, v in rec["texts"].items()}
    out.append(rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "texts"}), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
print("DONE", flush=True)
