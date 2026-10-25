"""Idle agent sessions: more long sessions than fit in GPU KV, with tool think time between turns.

Each session is a slice of one recorded long trajectory (results/agentic/trajectories_long_production.json),
replayed token-for-token under its own cache_salt: turn 0 of the slice is a cold prefill of the whole history
(60-120k tokens, it "arrives" with its context), turns 1..T-1 are the following recorded turns. Between turns
the session sleeps a think time drawn uniformly from [lo, hi] s (tool execution). Sessions start staggered.

usage: idle_bench.py <out.json> [--sessions 64] [--turns 6] [--think 5,30] [--start 60000,120000]
                     [--stagger 3] [--seed 0] [--tag name]
With --duration the run is a fixed time window (fair when an overloaded config cannot finish all turns).
Per later turn it records TTFT, cached tokens (local + CPU), and the recompute size; globally the delta of the
prefix-cache / external-cache / offload metrics, preemptions, and a 2 s time series of GPU KV usage.
"""
import argparse, json, os, random, re, statistics, sys, threading, time

from common import LONG_TRAJ, URL, complete, metrics, pct

p = argparse.ArgumentParser()
p.add_argument("out")
p.add_argument("--sessions", type=int, default=64)
p.add_argument("--turns", type=int, default=6)
p.add_argument("--think", default="5,30")
p.add_argument("--start", default="60000,120000")
p.add_argument("--stagger", type=float, default=3.0)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--tag", default="idle")
p.add_argument("--duration", type=float, default=0,
               help="measurement window in s (0 = run all turns). At the deadline the summary is written with the "
                    "turns still waiting for their first token counted as censored (TTFT >= time waited so far)")
a = p.parse_args()
think_lo, think_hi = map(float, a.think.split(","))
start_lo, start_hi = map(int, a.start.split(","))

trajs = json.load(open(LONG_TRAJ))["trajectories"]
rng = random.Random(a.seed)
sessions = []
for i in range(a.sessions):
    tr = trajs[i % len(trajs)]
    turns = tr["turns"]
    cands = [j for j in range(len(turns) - a.turns + 1) if start_lo <= len(turns[j]["prompt_ids"]) <= start_hi]
    if not cands:  # trajectory never reaches the window: take its last slice
        cands = [len(turns) - a.turns]
    j = rng.choice(cands)
    thinks = [rng.uniform(think_lo, think_hi) for _ in range(a.turns - 1)]
    sessions.append({"sid": i, "task": tr["task"], "start_turn": j, "turns": turns[j:j + a.turns], "thinks": thinks,
                     "salt": f"{a.tag}-{a.seed}-s{i}"})
tot0 = sum(len(s["turns"][0]["prompt_ids"]) for s in sessions)
print(f"{a.sessions} sessions x {a.turns} turns, first-turn context total {tot0/1e6:.2f}M tokens, "
      f"max {max(len(s['turns'][-1]['prompt_ids']) for s in sessions)}", flush=True)

res, lock, T0 = [], threading.Lock(), time.time()
inflight = {}  # sid -> {"turn", "t_start", "ttft"}
jl = open(a.out + "l", "w")  # per-turn records as they complete (<out>.jsonl)
samples, stop = [], threading.Event()


def sampler():
    while not stop.is_set():
        try:
            m = metrics()
            samples.append({"t": time.time() - T0, "kv_usage": m.get("vllm:kv_cache_usage_perc"),
                            "running": m.get("vllm:num_requests_running"), "waiting": m.get("vllm:num_requests_waiting")})
        except Exception:
            pass
        stop.wait(2.0)


def run_session(s):
    time.sleep(s["sid"] * a.stagger)
    prev_len = None
    for k, t in enumerate(s["turns"]):
        if k > 0:
            time.sleep(s["thinks"][k - 1])
        ts = time.time() - T0
        fl = {"turn": k, "t_start": ts, "ttft": None, "prompt": len(t["prompt_ids"])}
        with lock:
            inflight[s["sid"]] = fl
        r = complete(t["prompt_ids"], s["salt"], t["completion_tokens"], min_tokens=t["completion_tokens"],
                     ignore_eos=True, on_first=lambda x: fl.__setitem__("ttft", x))
        L = len(t["prompt_ids"])
        rec = {"sid": s["sid"], "task": s["task"], "turn": k, "t_start": ts, "prompt": L,
               "new_tokens": L - prev_len if prev_len else L, "prev_len": prev_len, "cached": r["cached"],
               "recompute": L - (r["cached"] or 0), "completion": t["completion_tokens"], "ttft": r["ttft"],
               "e2e": r["e2e"], "think_before": s["thinks"][k - 1] if k else 0.0}
        with lock:
            res.append(rec)
            inflight.pop(s["sid"], None)
            jl.write(json.dumps(rec) + "\n"); jl.flush()
        prev_len = L + t["completion_tokens"]


m0 = metrics()
th_s = threading.Thread(target=sampler, daemon=True); th_s.start()
ths = [threading.Thread(target=run_session, args=(s,), daemon=True) for s in sessions]
[t.start() for t in ths]
while any(t.is_alive() for t in ths) and not (a.duration and time.time() - T0 >= a.duration):
    time.sleep(min(15, max(0.1, (a.duration - (time.time() - T0)) if a.duration else 15)))
    with lock:
        n = len(res)
    print(f"  t={time.time()-T0:6.0f}s turns done {n}/{a.sessions*a.turns} in flight {len(inflight)}", flush=True)
wall = time.time() - T0
with lock:
    res = list(res)
    censored = [{"sid": sid, **f, "waited": wall - f["t_start"]} for sid, f in inflight.items()]
stop.set(); m1 = metrics()

delta = {k: m1[k] - m0.get(k, 0.0) for k in m1
         if re.search(r"prefix_cache|kv_offload|preemption|prompt_tokens_total|generation_tokens_total|"
                      r"simple_kv_offload|request_prefill_kv_computed", k)}
later = [r for r in res if r["turn"] > 0]
# censored later turns still waiting for the first token enter the TTFT percentiles with their wait so far
cens_lt = [c["waited"] for c in censored if c["turn"] > 0 and c["ttft"] is None]
lt = [r["ttft"] for r in later] + cens_lt
rc = [r["recompute"] for r in later]
evicted = [r for r in later if r["recompute"] > 8192]
summ = {
    "tag": a.tag, "url": URL, "sessions": a.sessions, "turns": a.turns, "think": a.think, "start": a.start,
    "wall_s": wall, "turns_done": len(res),
    "first_turn_ctx_total_tokens": tot0,
    "later_turns": len(later), "duration": a.duration,
    "censored_inflight": len(censored), "censored_later_waiting": len(cens_lt),
    "censored_first_waiting": sum(1 for c in censored if c["turn"] == 0 and c["ttft"] is None),
    "later_turns_per_s": len(later) / wall,
    "ttft_later_p50_ms": 1000 * (pct(lt, .5) or 0), "ttft_later_p90_ms": 1000 * (pct(lt, .9) or 0),
    "ttft_later_p99_ms": 1000 * (pct(lt, .99) or 0), "ttft_later_mean_ms": 1000 * statistics.mean(lt or [0]),
    "ttft_first_mean_ms": 1000 * statistics.mean([r["ttft"] for r in res if r["turn"] == 0] or [0]),
    "later_recompute_mean": statistics.mean(rc or [0]), "later_recompute_p50": pct(rc, .5),
    "later_recompute_p90": pct(rc, .9),
    "later_turns_evicted_gt8k": len(evicted), "later_frac_evicted": len(evicted) / max(len(later), 1),
    "later_cached_frac": sum(r["cached"] or 0 for r in later) / max(sum(r["prompt"] for r in later), 1),
    "ttft_later_nonevicted_p50_ms": 1000 * (pct([r["ttft"] for r in later if r["recompute"] <= 8192], .5) or 0),
    "ttft_later_evicted_p50_ms": 1000 * (pct([r["ttft"] for r in evicted], .5) or 0),
    "out_tok_s": sum(r["completion"] for r in res) / wall,
    "computed_prompt_tok_s": sum(r["recompute"] for r in res) / wall,
    "later_computed_prompt_tokens": sum(rc),
    "gpu_prefix_hit_rate": delta.get("vllm:prefix_cache_hits_total", 0) / max(delta.get("vllm:prefix_cache_queries_total", 0), 1),
    "external_hit_tokens": delta.get("vllm:external_prefix_cache_hits_total", 0),
    "external_query_tokens": delta.get("vllm:external_prefix_cache_queries_total", 0),
    "total_hit_rate": (delta.get("vllm:prefix_cache_hits_total", 0) + delta.get("vllm:external_prefix_cache_hits_total", 0))
                      / max(delta.get("vllm:prefix_cache_queries_total", 0), 1),
    "preemptions": delta.get("vllm:num_preemptions_total", 0),
    "kv_usage_max": max((s["kv_usage"] or 0) for s in samples) if samples else None,
}
print("SUMMARY", json.dumps(summ), flush=True)
json.dump({"summary": summ, "metric_delta": delta, "turns": res, "censored": censored, "samples": samples,
           "sessions": [{k: v for k, v in s.items() if k != "turns"} | {"first_len": len(s["turns"][0]["prompt_ids"])}
                        for s in sessions]}, open(a.out, "w"))
jl.close()
os._exit(0)  # do not wait for in-flight requests after the deadline
