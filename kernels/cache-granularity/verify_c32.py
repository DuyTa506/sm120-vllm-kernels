"""Verifier: 32 agents on LONG contexts (the production pain point: 1.58 s TTFT p50 at 32 agents).

Each of the 8 recorded long sessions is replayed by 4 agents (separate cache salts) starting at the first turn whose
prompt is >= --start-ctx tokens (or 13 turns before the end), for --turns turns. The first request per agent is a
cold prefill of the whole context (excluded from TTFT stats); later turns measure the steady state with ~32 idle +
active long sessions sharing the pool.

usage: BENCH_URL=... python3 verify_c32.py <out.json> <tag> [--conc 32] [--start-ctx 100000] [--turns 12]
"""
import argparse, json, os, statistics, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_subset as rs  # noqa: E402

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/agentic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("tag")
    ap.add_argument("--conc", type=int, default=32); ap.add_argument("--start-ctx", type=int, default=100000)
    ap.add_argument("--turns", type=int, default=12)
    a = ap.parse_args()
    trajs = json.load(open(os.path.join(D, "trajectories_long_production.json")))["trajectories"]
    jobs = []
    for copy in range(-(-a.conc // len(trajs))):
        for tr in trajs:
            t = tr["turns"]
            s = next((i for i, x in enumerate(t) if len(x["prompt_ids"]) >= a.start_ctx), len(t) - a.turns - 1)
            s = min(s, len(t) - a.turns - 1)
            jobs.append((tr["task"] + 100 * copy, t[s:s + a.turns + 1], f"c32-{a.tag}-{int(time.time())}-{copy}"))
    jobs = jobs[:a.conc]
    res, lock = [], threading.Lock()
    h0, q0 = rs.ab.metrics(); t0 = time.time()

    def agent(task, turns, salt):
        prev = None
        for k, t in enumerate(turns):
            p = t["prompt_ids"]
            ttft, e2e, cached = rs.one_turn(p, max(t["completion_tokens"], 1), salt)
            new = len(p) - len(prev) if prev is not None else len(p)
            with lock:
                res.append({"task": task, "turn": k, "prompt": len(p), "cached": cached or 0,
                            "prefilled": len(p) - (cached or 0), "new": new, "ttft": ttft, "e2e": e2e})
            prev = p
    ths = [threading.Thread(target=agent, args=j) for j in jobs]
    [t.start() for t in ths]; [t.join() for t in ths]
    wall = time.time() - t0; h1, q1 = rs.ab.metrics()
    later = [r for r in res if r["turn"] > 0]
    tt = sorted(1000 * r["ttft"] for r in later)
    summ = {"tag": a.tag, "agents": len(jobs), "turns": len(res), "wall_s": round(wall, 1),
            "ctx_mean": round(statistics.mean(r["prompt"] for r in later)),
            "later_prefilled_mean": round(statistics.mean(r["prefilled"] for r in later), 1),
            "later_new_mean": round(statistics.mean(r["new"] for r in later), 1),
            "later_full_miss": sum(r["cached"] == 0 for r in later),
            "later_ttft_p50_ms": round(rs.pct(tt, .5), 1), "later_ttft_p90_ms": round(rs.pct(tt, .9), 1),
            "later_ttft_mean_ms": round(statistics.mean(tt), 1),
            "first_ttft_mean_ms": round(1000 * statistics.mean(r["ttft"] for r in res if r["turn"] == 0), 1),
            "metric_hit_rate": round((h1 - h0) / max(q1 - q0, 1), 4)}
    print("C32", json.dumps(summ), flush=True)
    json.dump({"summary": summ, "turns": res}, open(a.out, "w"))


if __name__ == "__main__":
    main()
