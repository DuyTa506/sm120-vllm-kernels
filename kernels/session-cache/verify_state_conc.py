"""Verifier: direct state check under CONCURRENCY and adversarial session patterns (DEBUGDUMP=1 server).

Extends state_check.py (sequential, one chain at a time) with:
  * all chains run concurrently (threads), so snapshots, adoptions, CoW copies and block reuse interleave;
  * FORK:   at turn F two requests with the SAME session_id both extend the retained state, run concurrently
            (A = full next prompt, B = next prompt minus its last 7 tokens). Both must adopt and both
            final states must be exact; a missing copy-on-write would make them corrupt each other.
  * ABORT:  at turn A a request adopts the state, is aborted after a few streamed tokens, then the same
            request is retried and must adopt the (unmodified) retained state again.
  * DIVERGE: at turn D the history is edited in the middle (one token replaced) with the same session_id:
            it must NOT adopt (cached < edit position); the chain then continues on the edited history,
            and the next turn must adopt from the state built on the edited history.
Every final state is compared with cold-prefill references (state_check_analyze.py, unchanged).

usage: verify_state_conc.py <traj.json> <cases.json> --sessions 0,1,.. --turns N
"""
import argparse, json, os, threading, time, urllib.request
import hot_vs_cold as hvc
import session_bench as sb

FORK_T, ABORT_T, DIV_T = 2, 3, 4


def refs_for(ext, n, tag, label):
    refs = []
    for m in (n - 2, n - 1, n, n + 1, n - 1, n):
        if m > len(ext) or m < 2:
            continue
        rsid = f"{tag}-ref-{label}-{m}-{len(refs)}"
        hvc.req(ext[:m], rsid, 1, session=True, natural=False, logprobs=False)
        refs.append({"sid": rsid, "m": m})
    return refs


def aborted_request(prompt, sid, max_tokens, ntok=4):
    body = {"model": "bench", "prompt": prompt, "temperature": 0, "stream": True, "max_tokens": max_tokens,
            "min_tokens": max_tokens, "ignore_eos": True, "return_token_ids": True, "cache_salt": sid,
            "session_id": sid}
    req = urllib.request.Request(sb.URL + "/v1/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=600)
    got = 0
    for line in r:
        if line.startswith(b"data: ") and b"token_ids" in line:
            got += 1
            if got >= ntok:
                break
    r.close()  # client disconnect -> server aborts the request
    return got


def chain(traj, s, args, tag, cases, lock):
    turns = traj["turns"][: args.turns]
    sid = f"{tag}-S-{s}"
    prompt = list(turns[0]["prompt_ids"])
    prev_n = None

    def record(kind, label, out, prm, tail, extra=None):
        seq = prm + out["tokens"]
        n = len(seq)
        ext = seq + tail[:1]
        refs = refs_for(ext, n, tag, label)
        c = {"session": label, "turn": turn_no[0], "kind": kind, "sid": sid, "n": n, "prev_n": prev_n,
             "cached": out["cached"], "prompt": len(prm), "refs": refs}
        if extra:
            c.update(extra)
        with lock:
            cases.append(c)
        print(json.dumps({k: v for k, v in c.items() if k != "refs"}), flush=True)
        return seq

    turn_no = [0]
    for k, t in enumerate(turns):
        turn_no[0] = k
        c = t["completion_tokens"]
        natural = k % 2 == 1
        mt = c + 64 if natural else c
        tail = []
        if k + 1 < len(turns):
            tail = turns[k + 1]["prompt_ids"][len(t["prompt_ids"]) + c:]
        extra = {}
        if k == DIV_T and prev_n:
            # edit one token in the middle of the already-retained history
            pos = prev_n // 2
            prompt = list(prompt)
            prompt[pos] = 13 if prompt[pos] != 13 else 198
            extra = {"edit_pos": pos}
            kind = "diverge"
        elif k == ABORT_T and prev_n:
            got = aborted_request(prompt, sid, 3000)  # long, so the disconnect lands mid-generation
            time.sleep(1.0)
            extra = {"aborted_tokens": got}
            kind = "abort_retry"
        elif k == FORK_T and prev_n and len(prompt) - prev_n > 10:
            kind = "fork"
        else:
            kind = "chain"
        if kind == "fork":
            pb = prompt[:-7]
            res = {}

            def go(key, p):
                res[key] = hvc.req(p, sid, mt, session=True, natural=natural, logprobs=False)
            th = [threading.Thread(target=go, args=("A", prompt)), threading.Thread(target=go, args=("B", pb))]
            [x.start() for x in th]; [x.join() for x in th]
            record("forkB", f"{s}-forkB", res["B"], pb, [])
            seq = record("forkA", f"{s}-k{k}", res["A"], prompt, tail)
        else:
            out = hvc.req(prompt, sid, mt, session=True, natural=natural, logprobs=False)
            seq = record(kind, f"{s}-k{k}", out, prompt, tail, extra)
        prev_n = len(seq)
        prompt = seq + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj"); ap.add_argument("out")
    ap.add_argument("--sessions", default="0"); ap.add_argument("--turns", type=int, default=6)
    args = ap.parse_args()
    trajs = {t["task"]: t for t in json.load(open(args.traj))["trajectories"]}
    tag = f"vchk-{os.getpid()}-{int(time.time())}"
    cases, lock = [], threading.Lock()
    th = [threading.Thread(target=chain, args=(trajs[int(s)], int(s), args, tag, cases, lock))
          for s in args.sessions.split(",")]
    [x.start() for x in th]; [x.join() for x in th]
    json.dump({"tag": tag, "cases": cases}, open(args.out, "w"))


if __name__ == "__main__":
    main()
