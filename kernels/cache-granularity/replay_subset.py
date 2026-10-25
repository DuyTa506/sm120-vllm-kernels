"""Replay a subset of the recorded agentic sessions and measure, per turn, how many prompt tokens the server
actually prefilled (prompt - cached_tokens from usage.prompt_tokens_details) next to the ideal (only the tokens
appended since the previous turn's prompt).

usage: BENCH_URL=http://127.0.0.1:8117 python3 replay_subset.py <traj.json> <out.json> \
          [--sessions 0,1,4,7] [--conc 1,8] [--max-turns N] [--cap K] [--think S]

  --sessions  recorded sessions to replay (default: all)
  --conc      agents in parallel; more agents than sessions replays salted copies (as bench/agentic_bench.py)
  --max-turns replay only the first N turns of each session
  --cap       cap completion length at K tokens (default: recorded length)
  --think     seconds an agent waits between turns (tool time)

Needs the server started with --enable-prompt-tokens-details (cached_tokens per request).
"""
import argparse, json, os, statistics, sys, threading, time, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../bench"))
import agentic_bench as ab  # noqa: E402

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8117")
ab.URL = URL


def one_turn(prompt, max_tokens, salt):
    body = {"model": "bench", "prompt": prompt, "max_tokens": max_tokens, "min_tokens": max_tokens,
            "ignore_eos": True, "temperature": 0, "stream": True, "cache_salt": salt,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(URL + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; cached = None
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            j = json.loads(line[6:])
            if j.get("choices") and ttft is None:
                ttft = time.time() - t0
            u = j.get("usage")
            if u and u.get("prompt_tokens_details"):
                cached = u["prompt_tokens_details"].get("cached_tokens")
    return ttft, time.time() - t0, cached


def agent(traj, salt, res, args):
    prev = None
    for k, t in enumerate(traj["turns"][: args.max_turns or None]):
        if k and args.think:
            time.sleep(args.think)
        p = t["prompt_ids"]
        n_out = min(t["completion_tokens"], args.cap) if args.cap else t["completion_tokens"]
        ttft, e2e, cached = one_turn(p, max(n_out, 1), salt)
        ideal = len(p) - len(prev) if prev is not None and p[: len(prev)] == prev else len(p)
        res.append({"task": traj["task"], "turn": k, "prompt": len(p), "cached": cached,
                    "prefilled": len(p) - (cached or 0), "new": ideal, "ttft": ttft, "e2e": e2e})
        prev = p


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(int(len(xs) * q), len(xs) - 1)] if xs else float("nan")


def summarize(res, wall, dh, dq, c):
    later = [r for r in res if r["turn"] > 0]
    tt = [1000 * r["ttft"] for r in later]
    return {"conc": c, "turns": len(res), "wall_s": round(wall, 1),
            "later_prefilled_mean": round(statistics.mean(r["prefilled"] for r in later), 1),
            "later_new_mean": round(statistics.mean(r["new"] for r in later), 1),
            "later_excess_mean": round(statistics.mean(r["prefilled"] - r["new"] for r in later), 1),
            "later_ttft_p50_ms": round(pct(tt, .5), 1), "later_ttft_p90_ms": round(pct(tt, .9), 1),
            "later_ttft_mean_ms": round(statistics.mean(tt), 1),
            "first_ttft_mean_ms": round(1000 * statistics.mean(r["ttft"] for r in res if r["turn"] == 0), 1),
            "metric_hit_rate": round(dh / max(dq, 1), 4),
            "metric_prefilled_per_turn": round((dq - dh) / max(len(res), 1), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj"); ap.add_argument("out")
    ap.add_argument("--sessions", default=""); ap.add_argument("--conc", default="1,8")
    ap.add_argument("--max-turns", type=int, default=0); ap.add_argument("--cap", type=int, default=0)
    ap.add_argument("--think", type=float, default=0.0); ap.add_argument("--tag", default="")
    args = ap.parse_args()
    trajs = json.load(open(args.traj))["trajectories"]
    if args.sessions:
        keep = {int(x) for x in args.sessions.split(",")}
        trajs = [t for t in trajs if t["task"] in keep]
    report = {"args": vars(args)}
    for c in [int(x) for x in args.conc.split(",")]:
        copies = -(-c // len(trajs))
        base = f"cg-{args.tag}-c{c}-{os.getpid()}-{int(time.time())}"
        queue = [(dict(tr, task=k * 100 + tr["task"]), f"{base}-copy{k}") for k in range(copies) for tr in trajs]
        res, lock = [], threading.Lock()
        h0, q0 = ab.metrics(); t0 = time.time()

        def worker():
            while True:
                with lock:
                    if not queue:
                        return
                    tr, salt = queue.pop(0)
                agent(tr, salt, res, args)

        ths = [threading.Thread(target=worker) for _ in range(c)]
        [th.start() for th in ths]; [th.join() for th in ths]
        wall = time.time() - t0; h1, q1 = ab.metrics()
        s = summarize(res, wall, h1 - h0, q1 - q0, c)
        s["tag"] = args.tag
        print("SUMMARY", json.dumps(s), flush=True)
        report[c] = {"summary": s, "turns": res}
    json.dump(report, open(args.out, "w"))


if __name__ == "__main__":
    main()
