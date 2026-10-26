"""Verifier: output-level hot vs cold on FULL natural-stop generations, with a cold-vs-cold noise floor.

Per session chain (turn k+1 = prompt k + tokens generated in turn k + recorded tool output), every turn k>=1:
  hot   : session_id set (adopts the exact-position state of turn k-1)
  cold1 : same prompt, fresh salt, no session (full prefill)
  cold2 : same prompt, another fresh salt (noise floor: two identical cold runs)
  forkB : (turn FORK only) concurrently with hot, a second same-session request whose prompt is the hot
          prompt minus its last 7 tokens; compared with its own cold run.
All requests: temperature 0, max_tokens = recorded completion + 64, stop at EOS, logprobs=5.
Reports greedy LCP (common-prefix length) and completion length, hot-vs-cold1 next to cold2-vs-cold1, and
first-token top-5 TV. Sessions run concurrently with --conc.

usage: verify_hvc_full.py <traj.json> <out.json> --sessions 0,1 --turns N [--conc C]
"""
import argparse, json, math, os, threading, time
import hot_vs_cold as hvc

FORK_T = 3


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def tv(a, b):
    return hvc.tv(a, b)


def run(prompt, salt, mt, session):
    return hvc.req(prompt, salt, mt, session=session, natural=True, logprobs=True)


def chain(traj, s, args, tag, rows, lock):
    turns = traj["turns"][: args.turns]
    sid = f"{tag}-S{s}"
    prompt = list(turns[0]["prompt_ids"])
    prev_n = None
    for k, t in enumerate(turns):
        mt = t["completion_tokens"] + 64
        fork = {}
        if k == FORK_T and prev_n and len(prompt) - prev_n > 10:
            pb = prompt[:-7]
            th = threading.Thread(target=lambda: fork.update(out=run(pb, sid, mt, True)))
            th.start()
            hot = run(prompt, sid, mt, True)
            th.join()
        else:
            hot = run(prompt, sid, mt, True)
        gen = hot["tokens"]
        if k >= 1:
            c1 = run(prompt, f"{tag}-c1-{s}-{k}", mt, False)
            c2 = run(prompt, f"{tag}-c2-{s}-{k}", mt, False)
            row = {"session": s, "turn": k, "prompt": len(prompt), "prev_n": prev_n,
                   "hot_cached": hot["cached"], "adopted": hot["cached"] >= prev_n - 1,
                   "len_hot": len(gen), "len_c1": len(c1["tokens"]), "len_c2": len(c2["tokens"]),
                   "lcp_hot_c1": lcp(gen, c1["tokens"]), "lcp_c2_c1": lcp(c2["tokens"], c1["tokens"]),
                   "tv_hot_c1": tv(hot["top_logprobs"][0], c1["top_logprobs"][0]),
                   "tv_c2_c1": tv(c2["top_logprobs"][0], c1["top_logprobs"][0]),
                   "top1_hot_c1": gen[:1] == c1["tokens"][:1], "top1_c2_c1": c2["tokens"][:1] == c1["tokens"][:1]}
            if fork:
                fb = fork["out"]
                cb = run(pb, f"{tag}-cb-{s}-{k}", mt, False)
                row.update(forkB_cached=fb["cached"], forkB_adopted=fb["cached"] >= prev_n - 1,
                           lcp_forkB_cold=lcp(fb["tokens"], cb["tokens"]), len_forkB=len(fb["tokens"]),
                           len_forkB_cold=len(cb["tokens"]),
                           tv_forkB_cold=tv(fb["top_logprobs"][0], cb["top_logprobs"][0]))
            with lock:
                rows.append(row)
            print(json.dumps(row), flush=True)
        if k + 1 < len(turns):
            nxt = turns[k + 1]["prompt_ids"]
            tail = nxt[len(t["prompt_ids"]) + t["completion_tokens"]:]
            if gen and gen[-1] != 248046:
                gen = gen + [248046]
            prev_n = len(prompt) + len(hot["tokens"])
            prompt = prompt + gen + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj"); ap.add_argument("out")
    ap.add_argument("--sessions", default="0"); ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--conc", type=int, default=1)
    args = ap.parse_args()
    trajs = {t["task"]: t for t in json.load(open(args.traj))["trajectories"]}
    tag = f"vhvc-{os.getpid()}-{int(time.time())}"
    rows, lock = [], threading.Lock()
    todo = [int(x) for x in args.sessions.split(",")]

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                s = todo.pop(0)
            chain(trajs[s], s, args, tag, rows, lock)
    th = [threading.Thread(target=worker) for _ in range(args.conc)]
    [x.start() for x in th]; [x.join() for x in th]
    med = lambda v: sorted(v)[len(v) // 2] if v else None
    summ = {"cases": len(rows), "adopted": sum(r["adopted"] for r in rows),
            "top1_hot_c1": sum(r["top1_hot_c1"] for r in rows), "top1_c2_c1": sum(r["top1_c2_c1"] for r in rows),
            "mean_tv_hot_c1": sum(r["tv_hot_c1"] for r in rows) / max(1, len(rows)),
            "mean_tv_c2_c1": sum(r["tv_c2_c1"] for r in rows) / max(1, len(rows)),
            "median_lcp_hot_c1": med([r["lcp_hot_c1"] for r in rows]),
            "median_lcp_c2_c1": med([r["lcp_c2_c1"] for r in rows]),
            "full_match_hot_c1": sum(r["lcp_hot_c1"] == r["len_hot"] == r["len_c1"] for r in rows),
            "full_match_c2_c1": sum(r["lcp_c2_c1"] == r["len_c2"] == r["len_c1"] for r in rows),
            "mean_len_hot": sum(r["len_hot"] for r in rows) / max(1, len(rows)),
            "mean_len_c1": sum(r["len_c1"] for r in rows) / max(1, len(rows)),
            "fork_rows": [{k: r[k] for k in r if k.startswith(("forkB", "lcp_forkB", "len_forkB", "tv_forkB"))}
                          for r in rows if "forkB_cached" in r]}
    print("SUMMARY", json.dumps(summ), flush=True)
    json.dump({"summary": summ, "rows": rows}, open(args.out, "w"))


if __name__ == "__main__":
    main()
