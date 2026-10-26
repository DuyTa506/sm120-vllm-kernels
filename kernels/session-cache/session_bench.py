"""Replay recorded agentic sessions and measure per-turn prefill tokens and TTFT.

usage: session_bench.py <traj.json> <out.json> [--sessions 0,1] [--start J] [--turns N] [--conc C] [--copies K]
                        [--session-id] [--consistent] [--think S] [--tag T] [--natural-stop]

  --session-id    send session_id (= the per-agent cache salt) so the session state cache can act
  --consistent    build turn k+1 from the prompt and the tokens *generated* in turn k plus the recorded
                  tool output (what a real agent loop sends); default replays the recorded prompts
  --natural-stop  let the model stop at EOS (max_tokens = recorded + 64) instead of ignore_eos
  --think S       sleep S seconds between turns of a session (idle tool time)

Prefill tokens per turn = prompt_tokens - usage.prompt_tokens_details.cached_tokens (server needs
--enable-prompt-tokens-details). BENCH_URL selects the server.
"""
import argparse, json, os, statistics, threading, time, urllib.request

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8117")
IM_END = 248046


def post_stream(body):
    req = urllib.request.Request(URL + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    r = urllib.request.urlopen(req, timeout=1800)
    ttft, toks, usage, lps = None, [], None, []
    for line in r:
        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
            continue
        j = json.loads(line[6:])
        if j.get("usage"):
            usage = j["usage"]
        for ch in j.get("choices") or []:
            ids = ch.get("token_ids") or []
            if ids and ttft is None:
                ttft = time.time() - t0
            toks.extend(ids)
            lp = ch.get("logprobs")
            if lp and lp.get("top_logprobs"):
                lps.extend(lp["top_logprobs"])
    return {"ttft": ttft, "e2e": time.time() - t0, "tokens": toks, "usage": usage, "top_logprobs": lps}


def metrics():
    m = urllib.request.urlopen(URL + "/metrics", timeout=30).read().decode()
    out = {}
    for name in ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
                 "vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_draft_tokens_total",
                 "vllm:num_preemptions_total",
                 # engine steps, and server-side prefill time (first scheduled -> first token)
                 "vllm:iteration_tokens_total_count", "vllm:request_prefill_time_seconds_sum",
                 "vllm:request_prefill_time_seconds_count"):
        out[name] = sum(float(l.split()[-1]) for l in m.splitlines() if l.startswith(name))
    return out


def run_session(traj, salt, args, res):
    turns = traj["turns"][args.start:]
    turns = turns[: args.turns] if args.turns else turns
    prompt = list(turns[0]["prompt_ids"])
    for k, t in enumerate(turns):
        c = t["completion_tokens"]
        body = {"model": "bench", "prompt": prompt, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}, "return_token_ids": True, "cache_salt": salt}
        if args.natural_stop:
            body["max_tokens"] = c + 64
        else:
            body.update(max_tokens=c, min_tokens=c, ignore_eos=True)
        if args.session_id:
            body["session_id"] = salt
        out = post_stream(body)
        u = out["usage"] or {}
        cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
        gen = out["tokens"]
        rec = {"task": traj["task"], "salt": salt, "turn": k, "prompt": len(prompt), "cached": cached,
               "prefill": len(prompt) - cached, "completion": len(gen), "ttft": out["ttft"], "e2e": out["e2e"],
               "end": time.time()}
        if k + 1 < len(turns):
            nxt = turns[k + 1]["prompt_ids"]
            base = len(t["prompt_ids"])
            # does the recorded next prompt extend (this prompt + generated tokens)?
            rec["recorded_extends_generated"] = (len(prompt) == base and nxt[base:base + len(gen)] == gen)
            seg = nxt[base:base + c]
            lcp = 0
            while lcp < min(len(seg), len(gen)) and seg[lcp] == gen[lcp]:
                lcp += 1
            rec["gen_lcp"], rec["gen_len"], rec["rec_len"] = lcp, len(gen), c
            if lcp < min(len(seg), len(gen)):
                rec["diverge"] = {"rec": seg[lcp:lcp + 6], "gen": gen[lcp:lcp + 6]}
            new_prev = len(nxt) - base  # recorded completion + tool output + template
            rec["new_tokens"] = new_prev
            if args.consistent:
                tail = nxt[base + c:]
                prompt = prompt + gen + tail
            else:
                prompt = list(nxt)
        res.append(rec)
        if args.think and k + 1 < len(turns):
            time.sleep(args.think)


def summarize(res):
    later = [r for r in res if r["turn"] > 0]
    if not later:
        return {}
    tt = sorted(r["ttft"] for r in later)
    pf = [r["prefill"] for r in later]
    return {"turns": len(res), "later_turns": len(later),
            "prefill_mean": statistics.mean(pf), "prefill_median": statistics.median(pf),
            "prefill_p90": sorted(pf)[int(0.9 * len(pf))],
            "ttft_mean_ms": 1000 * statistics.mean(tt), "ttft_p50_ms": 1000 * tt[len(tt) // 2],
            "ttft_p90_ms": 1000 * tt[int(0.9 * len(tt))],
            "first_turn_ttft_ms": 1000 * statistics.mean(r["ttft"] for r in res if r["turn"] == 0),
            "recorded_extends_generated": sum(bool(r.get("recorded_extends_generated")) for r in res)
            / max(1, sum("recorded_extends_generated" in r for r in res))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj"); ap.add_argument("out")
    ap.add_argument("--sessions", default=""); ap.add_argument("--turns", type=int, default=0)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--conc", type=int, default=1); ap.add_argument("--copies", type=int, default=1)
    ap.add_argument("--session-id", action="store_true"); ap.add_argument("--consistent", action="store_true")
    ap.add_argument("--natural-stop", action="store_true"); ap.add_argument("--think", type=float, default=0)
    ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    trajs = json.load(open(args.traj))["trajectories"]
    if args.sessions:
        keep = {int(x) for x in args.sessions.split(",")}
        trajs = [t for t in trajs if t["task"] in keep]
    base = f"{args.tag}-{os.getpid()}-{int(time.time())}"
    queue = [(tr, f"{base}-s{tr['task']}-c{k}") for k in range(args.copies) for tr in trajs]
    m0 = metrics(); t0 = time.time(); res = []; lock = threading.Lock()

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                tr, salt = queue.pop(0)
            run_session(tr, salt, args, res)

    ths = [threading.Thread(target=worker) for _ in range(args.conc)]
    [t.start() for t in ths]; [t.join() for t in ths]
    m1 = metrics()
    summ = summarize(res)
    summ.update({"tag": args.tag, "conc": args.conc, "wall_s": time.time() - t0, "session_id": args.session_id,
                 "consistent": args.consistent, "natural_stop": args.natural_stop, "think": args.think,
                 "metrics_delta": {k: m1[k] - m0[k] for k in m0}})
    d = summ["metrics_delta"]
    if d["vllm:spec_decode_num_draft_tokens_total"]:
        summ["spec_accept_rate"] = d["vllm:spec_decode_num_accepted_tokens_total"] / d["vllm:spec_decode_num_draft_tokens_total"]
    print(json.dumps(summ), flush=True)
    json.dump({"summary": summ, "turns": res}, open(args.out, "w"))


if __name__ == "__main__":
    main()
