"""CPU-restore cost and correctness on real agentic continuations.

For each recorded pair (A = prompt of turn j, B = prompt of turn j+1, B extends A) of a long trajectory, with
A near each target length, three paths are run with separate cache salts:
  cold : B alone                                  (no cache)
  gpu  : A, then B                                (B hits A's state in the GPU prefix cache)
  cpu  : A, wait, reset the GPU prefix cache only (POST /reset_prefix_cache, needs VLLM_SERVER_DEV_MODE=1),
         then B                                   (B must restore A's state from host RAM)
Each B generates 32 greedy tokens with top-20 logprobs. Reports TTFT per path, cached tokens, offload load
bytes/time deltas (PCIe), first-token top-20 total-variation distance and greedy-text agreement between paths.
With --no-offload (baseline server) the cpu path degenerates to a recompute and is reported as such.

usage: pcie_correct.py <out.json> [--targets 25000,50000,100000,120000] [--reps 2] [--tag x]
"""
import argparse, json, math, time, urllib.error

from common import LONG_TRAJ, complete, metrics, post

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--targets", default="25000,50000,100000,120000")
p.add_argument("--reps", type=int, default=2)
p.add_argument("--tag", default="pc")
a = p.parse_args()

trajs = json.load(open(LONG_TRAJ))["trajectories"]
pairs = []
for tr in trajs:
    for j in range(len(tr["turns"]) - 1):
        pairs.append((len(tr["turns"][j]["prompt_ids"]), tr["task"], j))
cases = []
for tgt in map(int, a.targets.split(",")):
    used = set()
    for rep in range(a.reps):
        # nearest pair to the target from a trajectory not used yet for this target
        L, task, j = min((x for x in pairs if x[1] not in used), key=lambda x: abs(x[0] - tgt))
        used.add(task)
        tr = trajs[task]["turns"]
        cases.append({"target": tgt, "task": task, "turn": j, "A": tr[j]["prompt_ids"], "B": tr[j + 1]["prompt_ids"]})


def reset_gpu_cache():
    for _ in range(100):
        try:
            if post("/reset_prefix_cache").get("success"):
                return True
        except urllib.error.HTTPError as e:
            print("reset_prefix_cache failed:", e, "(server needs VLLM_SERVER_DEV_MODE=1)")
            return False
        time.sleep(0.2)
    return False


def tv(x, y):
    if not x or not y:
        return None
    x, y = x[0], y[0]
    keys = set(x) | set(y)
    return 0.5 * sum(abs(math.exp(x.get(k, -30)) - math.exp(y.get(k, -30))) for k in keys)


def top1(x):
    return max(x[0], key=x[0].get) if x else None


def run(prompt, salt):
    return complete(prompt, salt, 32, logprobs=20)


OFF = ["vllm:kv_offload_load_bytes_total", "vllm:kv_offload_load_time_total",
       "vllm:kv_offload_store_bytes_total", "vllm:kv_offload_store_time_total",
       "vllm:external_prefix_cache_hits_total", "vllm:prefix_cache_hits_total"]
run([1, 2, 3] * 100, f"{a.tag}-warm")  # warm-up
out = []
for ci, c in enumerate(cases):
    s = f"{a.tag}-{ci}"
    cold = run(c["B"], f"{s}-cold")
    cold2 = run(c["B"], f"{s}-cold2")  # determinism reference: an identical cold run
    complete(c["A"], f"{s}-gpu", 1)
    gpu = run(c["B"], f"{s}-gpu")
    complete(c["A"], f"{s}-cpu", 1)
    time.sleep(1.0)  # let the write-through stores of A finish
    ok = reset_gpu_cache()
    m0 = metrics()
    cpu = run(c["B"], f"{s}-cpu")
    time.sleep(0.5)
    m1 = metrics()
    d = {k.replace("vllm:", ""): m1.get(k, 0) - m0.get(k, 0) for k in OFF}
    rec = {"target": c["target"], "task": c["task"], "turn": c["turn"], "lenA": len(c["A"]), "lenB": len(c["B"]),
           "reset_ok": ok,
           "ttft_ms": {"cold": 1000 * cold["ttft"], "cold2": 1000 * cold2["ttft"], "gpu": 1000 * gpu["ttft"], "cpu": 1000 * cpu["ttft"]},
           "cached": {"cold": cold["cached"], "gpu": gpu["cached"], "cpu": cpu["cached"]},
           "cpu_path_metrics": d,
           "load_GBps": (d["kv_offload_load_bytes_total"] / d["kv_offload_load_time_total"] / 1e9
                         if d["kv_offload_load_time_total"] else None),
           "tv_top20": {"cold_cold2": tv(cold["top_logprobs"], cold2["top_logprobs"]),
                        "cold_gpu": tv(cold["top_logprobs"], gpu["top_logprobs"]),
                        "cold_cpu": tv(cold["top_logprobs"], cpu["top_logprobs"]),
                        "gpu_cpu": tv(gpu["top_logprobs"], cpu["top_logprobs"])},
           "top1_agree": {"cold_cold2": top1(cold["top_logprobs"]) == top1(cold2["top_logprobs"]),
                          "cold_gpu": top1(cold["top_logprobs"]) == top1(gpu["top_logprobs"]),
                          "cold_cpu": top1(cold["top_logprobs"]) == top1(cpu["top_logprobs"]),
                          "gpu_cpu": top1(gpu["top_logprobs"]) == top1(cpu["top_logprobs"])},
           "text_equal": {"cold_cold2": cold["text"] == cold2["text"], "cold_gpu": cold["text"] == gpu["text"], "cold_cpu": cold["text"] == cpu["text"],
                          "gpu_cpu": gpu["text"] == cpu["text"]},
           "texts": {"cold": cold["text"], "cold2": cold2["text"], "gpu": gpu["text"], "cpu": cpu["text"]}}
    out.append(rec)
    print(json.dumps({k: v for k, v in rec.items() if k != "texts"}), flush=True)
json.dump(out, open(a.out, "w"), indent=1)
