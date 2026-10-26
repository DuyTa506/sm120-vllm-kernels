"""Verifier: concurrent same-session adopters with DIFFERENT continuations (copy-on-write test), with controls.

Trial (session s, start turn j):
  base : prompt_j with session_id, ignore_eos, min(recorded completion, 300) tokens -> retained state at E
  fork trials (j even): A = base seq + real tool tail_j, B = base seq + tail of another session (different tokens),
         sent CONCURRENTLY with the same session_id; both must adopt. Without a private copy of the shared tail
         blocks one would overwrite the other's KV/state.
  solo trials (j odd):  A alone (control: ordinary adoption).
Each adopter is compared with a cold full prefill of its own prompt (fresh salt), and a second cold run gives the
noise floor. 24 greedy tokens (ignore_eos), top-20 logprobs: first-token TV and greedy LCP.

usage: verify_fork.py <traj.json> <out.json> --sessions 0,..,7 --starts 0,1,2,3,4,5
"""
import argparse, json, os, threading, time
import hot_vs_cold as hvc

N = 24


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def cmp(hot, cold):
    return {"tv": hvc.tv(hot["top_logprobs"][0], cold["top_logprobs"][0]), "lcp": lcp(hot["tokens"], cold["tokens"]),
            "top1": hot["tokens"][:1] == cold["tokens"][:1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj"); ap.add_argument("out")
    ap.add_argument("--sessions", default="0,1,2,3,4,5,6,7"); ap.add_argument("--starts", default="0,1,2,3,4,5")
    args = ap.parse_args()
    trajs = {t["task"]: t for t in json.load(open(args.traj))["trajectories"]}
    tag = f"vfork-{os.getpid()}-{int(time.time())}"
    ss = [int(x) for x in args.sessions.split(",")]
    rows = []
    for j in [int(x) for x in args.starts.split(",")]:
        for i, s in enumerate(ss):
            T = trajs[s]["turns"]
            if j + 1 >= len(T):
                continue
            sid = f"{tag}-{s}-{j}"
            c = min(T[j]["completion_tokens"], 300)
            base = hvc.req(list(T[j]["prompt_ids"]), sid, c, session=True, logprobs=False)
            seq = list(T[j]["prompt_ids"]) + base["tokens"]
            tailA = T[j + 1]["prompt_ids"][len(T[j]["prompt_ids"]) + T[j]["completion_tokens"]:]
            o = trajs[ss[(i + 1) % len(ss)]]["turns"]
            jo = min(j, len(o) - 2)
            tailB = o[jo + 1]["prompt_ids"][len(o[jo]["prompt_ids"]) + o[jo]["completion_tokens"]:]
            pA, pB = seq + tailA, seq + tailB
            mode = "fork" if j % 2 == 0 else "solo"
            res = {}
            if mode == "fork":
                def go(k, p):
                    res[k] = hvc.req(p, sid, N, session=True)
                th = [threading.Thread(target=go, args=("A", pA)), threading.Thread(target=go, args=("B", pB))]
                [x.start() for x in th]; [x.join() for x in th]
            else:
                res["A"] = hvc.req(pA, sid, N, session=True)
            row = {"session": s, "start": j, "mode": mode, "E_max": len(seq), "lenA": len(pA)}
            for k, p in (("A", pA), ("B", pB)):
                if k not in res:
                    continue
                c1 = hvc.req(p, f"{tag}-c1-{k}-{s}-{j}", N)
                c2 = hvc.req(p, f"{tag}-c2-{k}-{s}-{j}", N)
                row[f"{k}_cached"] = res[k]["cached"]
                row[f"{k}_adopted"] = res[k]["cached"] >= len(seq) - 1
                row[f"{k}_hot"] = cmp(res[k], c1)
                row[f"{k}_noise"] = cmp(c2, c1)
            rows.append(row)
            print(json.dumps(row), flush=True)

    def agg(mode, k, which):
        v = [r[f"{k}_{which}"] for r in rows if r["mode"] == mode and f"{k}_{which}" in r and r.get(f"{k}_adopted")]
        if not v:
            return None
        return {"n": len(v), "mean_tv": sum(x["tv"] for x in v) / len(v), "max_tv": max(x["tv"] for x in v),
                "tv_gt_0.4": sum(x["tv"] > 0.4 for x in v), "top1": sum(x["top1"] for x in v),
                "full24": sum(x["lcp"] == N for x in v), "median_lcp": sorted(x["lcp"] for x in v)[len(v) // 2]}
    summ = {f"{m}_{k}_{w}": agg(m, k, w) for m in ("fork", "solo") for k in ("A", "B") for w in ("hot", "noise")}
    summ["not_adopted"] = [(r["session"], r["start"], r["mode"]) for r in rows
                           if not r.get("A_adopted") or ("B_cached" in r and not r.get("B_adopted"))]
    print("SUMMARY", json.dumps(summ), flush=True)
    json.dump({"summary": summ, "rows": rows}, open(args.out, "w"))


if __name__ == "__main__":
    main()
