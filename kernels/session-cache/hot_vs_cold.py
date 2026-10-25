"""Correctness: outputs after a session-state hit vs a cold (no cache) run of the same prompt.

usage: hot_vs_cold.py <traj.json> <out.json> --sessions 0,1 --turns N [--cmp-tokens 24]

Per session, a chain of turns is run the way an agent loop would send it (turn k+1 = prompt k +
tokens generated in turn k + recorded tool output), with session_id set ("hot": each turn adopts the
exact-position state retained from the previous turn, including states that were themselves built on
an adopted state). Odd turns stop naturally at EOS (async scheduling then has a step in flight when the
request finishes, the offset-0 snapshot path); even turns run to max_tokens with ignore_eos.
Every turn k >= 1 is re-run twice with max_tokens = cmp-tokens and top-20 logprobs:
  cold:    fresh cache salt, full prefill
and compared with the hot output, next to a "normal" chain that sends the same prompts under its own
salt without session_id (stock aligned prefix hits): greedy token agreement over the first cmp-tokens tokens and the
total-variation distance between the first-token top-20 distributions. 'normal vs cold' is the noise
floor of an ordinary prefix-cache hit on this model; a wrong state shows up as TV ~ 0.5-1 and immediate
token disagreement.
"""
import argparse, json, math, os, time
import session_bench as sb


def tv(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(math.exp(a.get(k, -30)) - math.exp(b.get(k, -30))) for k in keys)


def agree(x, y):
    n = 0
    for p, q in zip(x, y):
        if p != q:
            break
        n += 1
    return n


def req(prompt, salt, max_tokens, session=False, natural=False, logprobs=True):
    body = {"model": "bench", "prompt": prompt, "temperature": 0, "stream": True, "max_tokens": max_tokens,
            "stream_options": {"include_usage": True}, "return_token_ids": True, "cache_salt": salt}
    if logprobs:
        body["logprobs"] = 20
    if not natural:
        body.update(min_tokens=max_tokens, ignore_eos=True)
    if session:
        body["session_id"] = salt
    out = sb.post_stream(body)
    u = out["usage"] or {}
    out["cached"] = ((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj"); ap.add_argument("out")
    ap.add_argument("--sessions", default="0"); ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--cmp-tokens", type=int, default=24)
    args = ap.parse_args()
    trajs = {t["task"]: t for t in json.load(open(args.traj))["trajectories"]}
    tag = f"hvc-{os.getpid()}-{int(time.time())}"
    rows = []
    for s in [int(x) for x in args.sessions.split(",")]:
        turns = trajs[s]["turns"][: args.turns]
        hot_salt, warm_salt = f"{tag}-hot-{s}", f"{tag}-warm-{s}"
        prompt = list(turns[0]["prompt_ids"])
        prev_len = None
        for k, t in enumerate(turns):
            c = t["completion_tokens"]
            natural = k % 2 == 1
            mt = c + 64 if natural else c
            hot = req(prompt, hot_salt, mt, session=True, natural=natural)
            # warm the stock path with the identical chain (no session_id)
            warm_chain = req(prompt, warm_salt, mt, session=False, natural=natural)
            gen = hot["tokens"]
            if k >= 1:
                n = min(args.cmp_tokens, len(gen))
                cold = req(prompt, f"{tag}-cold-{s}-{k}", n)
                # negative control: what an off-by-one state looks like (drop the last
                # token the session hit covered), cold
                cut = max(1, hot["cached"]) - 1
                neg = req(prompt[:cut] + prompt[cut + 1:], f"{tag}-neg-{s}-{k}", n)
                row = {"session": s, "turn": k, "prompt": len(prompt), "prev_len": prev_len,
                       "hot_cached": hot["cached"], "warm_cached": warm_chain["cached"],
                       "cold_cached": cold["cached"],
                       "agree_hot_cold": agree(gen[:n], cold["tokens"][:n]),
                       "agree_warm_cold": agree(warm_chain["tokens"][:n], cold["tokens"][:n]),
                       "n": n,
                       "tv_hot_cold": tv(hot["top_logprobs"][0], cold["top_logprobs"][0]),
                       "tv_warm_cold": tv(warm_chain["top_logprobs"][0], cold["top_logprobs"][0]),
                       "tv_neg_cold": tv(neg["top_logprobs"][0], cold["top_logprobs"][0]),
                       "agree_neg_cold": agree(neg["tokens"][:n], cold["tokens"][:n]),
                       "hot_ttft": hot["ttft"], "warm_ttft": warm_chain["ttft"], "natural_prev": (k - 1) % 2 == 1}
                rows.append(row)
                print(json.dumps(row), flush=True)
            if k + 1 < len(turns):
                nxt = turns[k + 1]["prompt_ids"]
                base = len(t["prompt_ids"])
                prev_len = len(prompt) + len(gen)  # tokens of the finished hot request
                # recorded tool output after the recorded completion; generated tokens replace it
                tail = nxt[base + c:]
                if natural and gen and gen[-1] != sb.IM_END:
                    gen = gen + [sb.IM_END]  # keep the chat structure if it did not stop at EOS
                prompt = prompt + gen + tail
                # keep the warm chain on the same prompts (it generated its own tokens; only the
                # prompts matter for its cache)
    summ = {
        "cases": len(rows),
        "hot_session_hits": sum(r["hot_cached"] > r["warm_cached"] for r in rows),
        "mean_tv_hot_cold": sum(r["tv_hot_cold"] for r in rows) / max(1, len(rows)),
        "max_tv_hot_cold": max((r["tv_hot_cold"] for r in rows), default=0),
        "mean_tv_warm_cold": sum(r["tv_warm_cold"] for r in rows) / max(1, len(rows)),
        "max_tv_warm_cold": max((r["tv_warm_cold"] for r in rows), default=0),
        "full_agree_hot_cold": sum(r["agree_hot_cold"] == r["n"] for r in rows),
        "full_agree_warm_cold": sum(r["agree_warm_cold"] == r["n"] for r in rows),
        "first_token_agree_hot_cold": sum(r["agree_hot_cold"] >= 1 for r in rows),
        "first_token_agree_warm_cold": sum(r["agree_warm_cold"] >= 1 for r in rows),
        "first_token_agree_neg_cold": sum(r["agree_neg_cold"] >= 1 for r in rows),
        "mean_tv_neg_cold": sum(r["tv_neg_cold"] for r in rows) / max(1, len(rows)),
        "full_agree_neg_cold": sum(r["agree_neg_cold"] == r["n"] for r in rows),
    }
    print("SUMMARY", json.dumps(summ), flush=True)
    json.dump({"summary": summ, "rows": rows}, open(args.out, "w"))


if __name__ == "__main__":
    main()
