"""Verifier: correctness of hot (GPU/CPU-restored) prefixes UNDER CONCURRENT LOAD and eviction pressure.

Run next to a background idle_bench.py load. Each probe session (own salt s):
  coldB  = B(fresh salt)          reference under load
  A(s)   (1 token)                prefill A, written through to CPU
  sleep T (60-200 s)              let the background load evict A from the GPU (and, with a small CPU tier, from CPU)
  hotB   = B(s)                   restored from GPU / CPU / partly recomputed
  coldB2 = B(fresh salt)          second reference (noise floor under load)
  [every 2nd probe] sleep, C(s) with C = A[:cut] + foreign tail (divergent), and C(fresh) reference
Compares first-token top-20 TV, top-1 and greedy 32-token text equality, and records cached tokens.
usage: verify_load.py <out.json> [--probes 16] [--stagger 20] [--seed 7] [--tag vl]
"""
import argparse, json, math, random, threading, time

from common import LONG_TRAJ, complete, metrics

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--probes", type=int, default=16)
p.add_argument("--stagger", type=float, default=20)
p.add_argument("--sleep", default="60,200")
p.add_argument("--len", default="20000,50000")
p.add_argument("--seed", type=int, default=7)
p.add_argument("--tag", default="vl")
a = p.parse_args()
lo, hi = map(float, a.sleep.split(","))
llo, lhi = map(int, a.len.split(","))

trajs = json.load(open(LONG_TRAJ))["trajectories"]
rng = random.Random(a.seed)
pairs = [(tr["task"], j) for tr in trajs for j in range(len(tr["turns"]) - 1)
         if llo <= len(tr["turns"][j]["prompt_ids"]) <= lhi
         and len(tr["turns"][j + 1]["prompt_ids"]) > len(tr["turns"][j]["prompt_ids"])]
probes = []
for i in range(a.probes):
    task, j = rng.choice(pairs)
    A, B = trajs[task]["turns"][j]["prompt_ids"], trajs[task]["turns"][j + 1]["prompt_ids"]
    assert B[:len(A)] == A
    ft = trajs[(task + 3) % 8]["turns"][-1]["prompt_ids"][-3000:]
    cut = int(len(A) * rng.uniform(0.3, 0.9)) + 333
    probes.append({"i": i, "task": task, "turn": j, "A": A, "B": B, "C": A[:cut] + ft, "cut": cut,
                   "sleep": rng.uniform(lo, hi), "sleep2": rng.uniform(lo, hi)})


def tv(x, y):
    if not x or not y:
        return None
    x, y = x[0], y[0]
    keys = set(x) | set(y)
    return 0.5 * sum(abs(math.exp(x.get(k, -30)) - math.exp(y.get(k, -30))) for k in keys)


def top1(x):
    return max(x[0], key=x[0].get) if x else None


def cmp(r, ref):
    return {"tv": tv(r["top_logprobs"], ref["top_logprobs"]), "top1": top1(r["top_logprobs"]) == top1(ref["top_logprobs"]),
            "text_eq": r["text"] == ref["text"]}


res, lock, T0 = [], threading.Lock(), time.time()


def go(pr):
    time.sleep(pr["i"] * a.stagger)
    s, f = f"{a.tag}-{pr['i']}-s", f"{a.tag}-{pr['i']}-f"
    rec = {k: pr[k] for k in ("i", "task", "turn", "cut", "sleep", "sleep2")}
    rec["lenA"], rec["lenB"], rec["lenC"] = len(pr["A"]), len(pr["B"]), len(pr["C"])
    try:
        coldB = complete(pr["B"], f + "B1", 32, logprobs=20)
        complete(pr["A"], s, 1)
        time.sleep(pr["sleep"])
        hotB = complete(pr["B"], s, 32, logprobs=20)
        coldB2 = complete(pr["B"], f + "B2", 32, logprobs=20)
        rec["B"] = {"cached_hot": hotB["cached"], "ttft_hot": hotB["ttft"], "ttft_cold": coldB["ttft"],
                    "hot_vs_cold": cmp(hotB, coldB), "cold2_vs_cold": cmp(coldB2, coldB),
                    "hot_vs_cold2": cmp(hotB, coldB2), "texts": [coldB["text"], coldB2["text"], hotB["text"]]}
        if pr["i"] % 2 == 0:
            time.sleep(pr["sleep2"])
            hotC = complete(pr["C"], s, 32, logprobs=20)
            coldC = complete(pr["C"], f + "C1", 32, logprobs=20)
            rec["C"] = {"cached_hot": hotC["cached"], "hot_vs_cold": cmp(hotC, coldC),
                        "texts": [coldC["text"], hotC["text"]]}
    except Exception as e:  # a crash/timeout is a finding, record it
        rec["error"] = repr(e)
    rec["t_end"] = time.time() - T0
    with lock:
        res.append(rec)
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("B", "C")} |
                     {"B": {k: v for k, v in rec.get("B", {}).items() if k != "texts"},
                      "C": {k: v for k, v in rec.get("C", {}).items() if k != "texts"}}), flush=True)


m0 = metrics()
ths = [threading.Thread(target=go, args=(pr,)) for pr in probes]
[t.start() for t in ths]
[t.join() for t in ths]
m1 = metrics()
delta = {k: m1[k] - m0.get(k, 0.0) for k in m1 if "offload" in k or "prefix_cache" in k or "preempt" in k or "spec_decode" in k}
json.dump({"probes": res, "metric_delta": delta}, open(a.out, "w"), indent=1)
print("METRICS", json.dumps(delta), flush=True)
print("DONE", flush=True)
