"""Verifier: direct state check focused on concurrent same-session requests (DEBUGDUMP=1 server).

Per session (chains run concurrently), turns:
  0,1     chain (adopted hits)
  2       FORK-early: A = next prompt (recorded length), B = A's prompt minus 7 tokens with ~1/3 of the tokens,
          concurrently, same session_id -> B finishes first, A finishes LAST, A's entry survives.
  3,4     chain from A: turn 3 must ADOPT A's retained state; its own final state is checked against cold refs,
          which is the end-to-end test of the state retained after a fork.
  5       FORK-late: B gets 3x the tokens -> B finishes last (A's entry is replaced by B's).
  6       chain from A (expected: miss, B's entry does not match) -> stock path.
Every final state is compared with cold-prefill references (state_check_analyze.py). Each case records the
wall time at which its response completed so finish order can be checked.

usage: verify_state_fork.py <traj.json> <cases.json> --sessions 0,..,5
"""
import argparse, json, os, threading, time
import hot_vs_cold as hvc
from verify_state_conc import refs_for


def chain(traj, s, tag, cases, lock):
    turns = traj["turns"][:7]
    sid = f"{tag}-S-{s}"
    prompt = list(turns[0]["prompt_ids"])
    prev_n = None

    def record(kind, k, out, prm, tail, tdone):
        seq = prm + out["tokens"]
        n = len(seq)
        refs = refs_for(seq + tail[:1], n, tag, f"{s}-{kind}-{k}")
        c = {"session": f"{s}-{kind}-{k}", "turn": k, "kind": kind, "sid": sid, "n": n, "prev_n": prev_n,
             "cached": out["cached"], "prompt": len(prm), "t_done": tdone, "refs": refs}
        with lock:
            cases.append(c)
        print(json.dumps({x: y for x, y in c.items() if x != "refs"}), flush=True)
        return seq

    for k, t in enumerate(turns):
        c = t["completion_tokens"]
        mt = max(c, 24)
        tail = turns[k + 1]["prompt_ids"][len(t["prompt_ids"]) + c:] if k + 1 < len(turns) else []
        if k in (2, 5) and prev_n and len(prompt) - prev_n > 10:
            pb = prompt[:-7]
            mtb = max(4, mt // 3) if k == 2 else mt * 3
            res, done = {}, {}

            def go(key, p, m):
                res[key] = hvc.req(p, sid, m, session=True, natural=False, logprobs=False)
                done[key] = time.time()
            th = [threading.Thread(target=go, args=("A", prompt, mt)), threading.Thread(target=go, args=("B", pb, mtb))]
            [x.start() for x in th]; [x.join() for x in th]
            kind = "forkEarly" if k == 2 else "forkLate"
            record(kind + "B", k, res["B"], pb, [], done["B"])
            seq = record(kind + "A", k, res["A"], prompt, tail, done["A"])
        else:
            out = hvc.req(prompt, sid, mt, session=True, natural=False, logprobs=False)
            kind = {3: "afterForkEarly", 6: "afterForkLate"}.get(k, "chain")
            seq = record(kind, k, out, prompt, tail, time.time())
        prev_n = len(seq)
        prompt = seq + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj"); ap.add_argument("out")
    ap.add_argument("--sessions", default="0,1,2,3,4,5")
    args = ap.parse_args()
    trajs = {t["task"]: t for t in json.load(open(args.traj))["trajectories"]}
    tag = f"vfst-{os.getpid()}-{int(time.time())}"
    cases, lock = [], threading.Lock()
    th = [threading.Thread(target=chain, args=(trajs[int(s)], int(s), tag, cases, lock))
          for s in args.sessions.split(",")]
    [x.start() for x in th]; [x.join() for x in th]
    json.dump({"tag": tag, "cases": cases}, open(args.out, "w"))


if __name__ == "__main__":
    main()
