"""Verifier: how far back does a request fall when the session entry MISSES after a chain of adoptions?

For each short session: run turns 0..T-1 as a consistent chain (ignore_eos, recorded completion lengths), either with
session_id (adopted hits) or without (stock / --prefix-match-unit path). Then send a "diverged" request: the last
prompt with one token edited `back` tokens before the end of the previous turn's sequence (what a client that
re-renders/trims the recent history sends). Same session_id, so the session entry is consulted and misses.
Reports the cached tokens of the diverged request: the fallback depth. With --prefix-match-unit 16 alone every
turn's prompt tail is checkpointed; if adopted turns do not save theirs, the session run falls back further.

usage: verify_fallback.py <traj.json> <out.json> --sessions 0,1,2,3 --turns 6 --back 50
"""
import argparse, json, os, time
import hot_vs_cold as hvc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj"); ap.add_argument("out")
    ap.add_argument("--sessions", default="0,1,2,3"); ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--back", type=int, default=50)
    args = ap.parse_args()
    trajs = {t["task"]: t for t in json.load(open(args.traj))["trajectories"]}
    tag = f"vfb-{os.getpid()}-{int(time.time())}"
    rows = []
    for mode in ("sess", "nosess"):
        for s in [int(x) for x in args.sessions.split(",")]:
            turns = trajs[s]["turns"][: args.turns]
            salt = f"{tag}-{mode}-{s}"
            prompt = list(turns[0]["prompt_ids"])
            cached = []
            for k, t in enumerate(turns):
                c = t["completion_tokens"]
                out = hvc.req(prompt, salt, c, session=(mode == "sess"), natural=False, logprobs=False)
                cached.append(out["cached"])
                prev_prompt = len(prompt)
                seq = prompt + out["tokens"]
                if k + 1 < len(turns):
                    tail = turns[k + 1]["prompt_ids"][len(t["prompt_ids"]) + c:]
                    prompt = seq + tail
            # diverge: edit one token `back` tokens before the end of the last finished sequence
            pos = len(seq) - args.back
            div = list(seq) + [13, 13, 13]
            div[pos] = 13 if div[pos] != 13 else 198
            out = hvc.req(div, salt, 4, session=(mode == "sess"), natural=False, logprobs=False)
            row = {"mode": mode, "session": s, "chain_cached": cached, "last_prompt": prev_prompt,
                   "seq_len": len(seq), "edit_pos": pos, "diverged_cached": out["cached"],
                   "lost": pos - out["cached"]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    json.dump({"rows": rows}, open(args.out, "w"))


if __name__ == "__main__":
    main()
