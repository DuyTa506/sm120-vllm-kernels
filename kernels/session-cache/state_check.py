"""Direct check of the snapshot position: compare retained states with cold-prefill reference states.

Needs a server started with DEBUGDUMP=1 (run_gpu.sh), i.e. the debug variant files in debug/ that log
every snapshot (E, token hash) and dump its state block for 3 GDN layers to /tmp/ssdump in the container.

usage: state_check.py <traj.json> <cases.json> --sessions 0,1,2 --turns 4

Per session a chain of turns runs with session_id (odd turns stop at EOS, even turns run to max_tokens
with ignore_eos), so snapshots come from decode/verify steps, with and without an in-flight step, and
after adopted hits. After each turn (sequence `seq`, n tokens) cold reference requests are sent with a
fresh salt and max_tokens=1 for prompts seq[:m], m in n-2..n+1 (m=n+1 appends the next turn's first new
token): each is snapshotted after its *prefill*, which is an independent path to the state after exactly
m (or m+1) tokens. state_check_analyze.py (run inside the container) pairs them by position and token
hash: the session snapshot at E must match the reference at E, and differ clearly from E-1 and E+1.
"""
import argparse, json, os, time
import hot_vs_cold as hvc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj"); ap.add_argument("out")
    ap.add_argument("--sessions", default="0"); ap.add_argument("--turns", type=int, default=4)
    args = ap.parse_args()
    trajs = {t["task"]: t for t in json.load(open(args.traj))["trajectories"]}
    tag = f"chk-{os.getpid()}-{int(time.time())}"
    cases = []
    for s in [int(x) for x in args.sessions.split(",")]:
        turns = trajs[s]["turns"][: args.turns]
        sid = f"{tag}-A-{s}"
        prompt = list(turns[0]["prompt_ids"])
        for k, t in enumerate(turns):
            c = t["completion_tokens"]
            natural = k % 2 == 1
            out = hvc.req(prompt, sid, c + 64 if natural else c, session=True, natural=natural,
                          logprobs=False)
            seq = prompt + out["tokens"]
            n = len(seq)
            tail = []
            if k + 1 < len(turns):
                nxt = turns[k + 1]["prompt_ids"]
                tail = nxt[len(t["prompt_ids"]) + c:]
            ext = seq + tail[:1]
            refs = []
            for m in (n - 2, n - 1, n, n + 1, n - 1, n):  # n-1 and n twice: prefill-vs-prefill noise
                if m > len(ext):
                    continue
                rsid = f"{tag}-ref-{s}-{k}-{m}-{len(refs)}"
                hvc.req(ext[:m], rsid, 1, session=True, natural=False, logprobs=False)
                refs.append({"sid": rsid, "m": m})
            cases.append({"session": s, "turn": k, "sid": sid, "n": n, "natural": natural,
                          "cached": out["cached"], "prompt": len(prompt), "refs": refs})
            print(json.dumps(cases[-1]), flush=True)
            prompt = seq + tail
    json.dump({"tag": tag, "cases": cases}, open(args.out, "w"))


if __name__ == "__main__":
    main()
