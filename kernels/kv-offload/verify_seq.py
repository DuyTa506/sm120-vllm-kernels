"""Verifier: sequential (idle GPU) correctness of native CPU offload restores, including paths the original
pcie_correct.py does not exercise.

Per case (A = recorded prompt of turn j, B = prompt of turn j+1 which extends A, C = A[:p] + foreign tail,
i.e. a DIVERGENT session that shares only part of A's history), all with one cache salt s unless noted:
  1. A(s)           cold reference for A (32 greedy tokens, top-20 logprobs); its prefill is written to CPU
  2. A(fresh)       cold2 for A (noise floor)
  3. reset; B(s)    B restored purely from CPU                      [cpuB]
  4. reset; C(s)    divergent prompt restored from an intermediate CPU checkpoint    [cpuC]
  5. reset; A(s)    exact retry of A restored from CPU               [cpuA]
  6. reset; C(s); B(s)  B with a partial GPU hit (C's prefix) extended from CPU      [mixB]
  7. B(fresh) x2, C(fresh) x2    cold references / noise for B and C
  8. A(g); B(g)     normal GPU prefix hit for B                       [gpuB]
"reset" = POST /reset_prefix_cache (GPU prefix cache only; needs VLLM_SERVER_DEV_MODE=1).
Each request records cached tokens, deltas of local and external prefix-cache hits, offload load bytes and
spec-decode accepted/draft tokens (requests are sequential, so global metric deltas are per request).
usage: verify_seq.py <out.json> [--targets 30000,60000,100000] [--reps 2] [--tag v]
"""
import argparse, json, math, time, urllib.error

from common import LONG_TRAJ, complete, metrics, post

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--targets", default="30000,60000,100000")
p.add_argument("--reps", type=int, default=2)
p.add_argument("--tag", default="v")
p.add_argument("--split", type=float, default=0.62, help="C keeps A[:split*len(A)] then a foreign tail")
a = p.parse_args()

trajs = json.load(open(LONG_TRAJ))["trajectories"]
pairs = [(len(tr["turns"][j]["prompt_ids"]), tr["task"], j) for tr in trajs for j in range(len(tr["turns"]) - 1)]
cases = []
for tgt in map(int, a.targets.split(",")):
    used = set()
    for rep in range(a.reps):
        L, task, j = min((x for x in pairs if x[1] not in used and len(trajs[x[1]]["turns"][x[2] + 1]["prompt_ids"]) > x[0]),
                         key=lambda x: abs(x[0] - tgt))
        used.add(task)
        tr = trajs[task]["turns"]
        A, B = tr[j]["prompt_ids"], tr[j + 1]["prompt_ids"]
        assert B[:len(A)] == A, "B must extend A"
        # foreign tail: 3000 tokens from a different trajectory (valid token ids, unrelated content)
        ft = trajs[(task + 3) % len(trajs)]["turns"][-1]["prompt_ids"][-3000:]
        cut = int(len(A) * a.split) + 777  # deliberately not block aligned
        C = A[:cut] + ft
        cases.append({"target": tgt, "task": task, "turn": j, "A": A, "B": B, "C": C, "cut": cut})

MET = ["vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total", "vllm:external_prefix_cache_hits_total",
       "vllm:kv_offload_load_bytes_total", "vllm:kv_offload_store_bytes_total",
       "vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_draft_tokens_total",
       "vllm:kv_offload_allocation_failure_total"]


def reset():
    time.sleep(1.0)  # let write-through stores finish
    for _ in range(100):
        try:
            if post("/reset_prefix_cache").get("success"):
                return True
        except urllib.error.HTTPError as e:
            print("reset failed", e)
            return False
        time.sleep(0.2)
    return False


def run(prompt, salt, n=32):
    m0 = metrics()
    r = complete(prompt, salt, n, logprobs=20)
    time.sleep(0.3)
    m1 = metrics()
    r["d"] = {k.replace("vllm:", ""): m1.get(k, 0) - m0.get(k, 0) for k in MET}
    r["len"] = len(prompt)
    return r


def tv(x, y):
    if not x or not y:
        return None
    x, y = x[0], y[0]
    keys = set(x) | set(y)
    return 0.5 * sum(abs(math.exp(x.get(k, -30)) - math.exp(y.get(k, -30))) for k in keys)


def top1(x):
    return max(x[0], key=x[0].get) if x else None


def cmp(r, ref):
    t = tv(r["top_logprobs"], ref["top_logprobs"])
    cp = 0
    for u, v in zip(r["text"], ref["text"]):
        if u != v:
            break
        cp += 1
    return {"tv": t, "top1": top1(r["top_logprobs"]) == top1(ref["top_logprobs"]), "text_eq": r["text"] == ref["text"],
            "common_prefix_chars": cp}


def slim(r):
    return {"ttft_ms": 1000 * r["ttft"], "cached": r["cached"], "len": r["len"], "d": r["d"], "text": r["text"]}


run([1, 2, 3] * 100, f"{a.tag}-warm")
out = []
for ci, c in enumerate(cases):
    s, f = f"{a.tag}-{ci}-s", f"{a.tag}-{ci}-f"
    R = {}
    R["coldA"] = run(c["A"], s)
    R["coldA2"] = run(c["A"], f + "A2")
    ok = [reset()]
    R["cpuB"] = run(c["B"], s)
    ok.append(reset())
    R["cpuC"] = run(c["C"], s)
    ok.append(reset())
    R["cpuA"] = run(c["A"], s)
    ok.append(reset())
    R["cpuC2"] = run(c["C"], s)
    R["mixB"] = run(c["B"], s)
    R["coldB"] = run(c["B"], f + "B1")
    R["coldB2"] = run(c["B"], f + "B2")
    R["coldC"] = run(c["C"], f + "C1")
    R["coldC2"] = run(c["C"], f + "C2")
    run(c["A"], f + "g", 1)
    R["gpuB"] = run(c["B"], f + "g")
    comps = {
        "A: cold2 vs cold (noise)": cmp(R["coldA2"], R["coldA"]),
        "A: cpu-retry vs cold": cmp(R["cpuA"], R["coldA"]),
        "B: cold2 vs cold (noise)": cmp(R["coldB2"], R["coldB"]),
        "B: gpu-hit vs cold": cmp(R["gpuB"], R["coldB"]),
        "B: cpu vs cold": cmp(R["cpuB"], R["coldB"]),
        "B: mixed gpu+cpu vs cold": cmp(R["mixB"], R["coldB"]),
        "C: cold2 vs cold (noise)": cmp(R["coldC2"], R["coldC"]),
        "C: cpu-divergent vs cold": cmp(R["cpuC"], R["coldC"]),
        "C: cpu-divergent(2nd) vs cold": cmp(R["cpuC2"], R["coldC"]),
    }
    rec = {"target": c["target"], "task": c["task"], "turn": c["turn"], "lenA": len(c["A"]), "lenB": len(c["B"]),
           "lenC": len(c["C"]), "cut": c["cut"], "reset_ok": all(ok), "cmp": comps,
           "runs": {k: slim(v) for k, v in R.items()}}
    out.append(rec)
    print(json.dumps({"case": ci, "lenA": rec["lenA"], "cut": c["cut"],
                      "cached": {k: v["cached"] for k, v in R.items()},
                      "ext": {k: v["d"]["external_prefix_cache_hits_total"] for k, v in R.items()},
                      "tv": {k: (round(v["tv"], 3) if v["tv"] is not None else None, v["top1"], v["text_eq"]) for k, v in comps.items()}}),
          flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
print("DONE", flush=True)
