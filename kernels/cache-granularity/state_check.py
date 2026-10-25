"""Engine-level hot-vs-cold check of the GDN recurrent state (needs the dump hook, see make_dump_build.py).

The end-to-end check (hot_cold.py) cannot resolve small state errors on this server: two identical cold runs already
differ by ~0.1 total variation in the first-token distribution (the server is not bit-deterministic). The input of
GDN layer 0 is the token embedding, so its state after a prefill depends only on the tokens and on the chunking of the
recurrence. Comparing that state between a hot prefill (resumed from a prefix-cache Mamba checkpoint) and a cold
prefill of the same prompt therefore isolates exactly the property that matters: the restored checkpoint is the
state of the right position. A checkpoint that is off by one 16-token hash unit is emulated by a cold run of the same
prompt with 16 tokens removed before the hit position (drop16), which gives the error scale to compare against.

For each recorded (session, turn k) pair: cold1(turn k), cold2(turn k), drop16, warm(turn k-1, 64 decode tokens),
hot(turn k). Reports relative L2 error of the final layer-0 (and layer-2) recurrent state and the max abs error of
the conv state, each against cold1.

usage: BENCH_URL=... python3 state_check.py <dump_dir> <out.json> <tag>
"""
import glob, json, os, sys, time, urllib.request

import torch

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8117")
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/agentic")
SHORT = [(s, k) for s in range(8) for k in (3, 9, 15)]
LONG = [(0, 20), (4, 60), (7, 50), (2, 100), (5, 110), (1, 40)]


def post(body):
    r = urllib.request.Request(URL + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())


def run(dump, prompt, salt, n=1):
    before = set(os.listdir(dump))
    j = post({"model": "bench", "prompt": prompt, "max_tokens": n, "min_tokens": n, "ignore_eos": True,
              "temperature": 0, "cache_salt": salt})
    time.sleep(0.2)
    new = sorted(set(os.listdir(dump)) - before - {"ENABLE"})
    last = {}
    for f in new:  # <counter>_<layer>.pt, increasing counter: keep the final prefill chunk per layer
        last[f.split("_")[1][:-3]] = f
    out = {}
    for layer, f in last.items():
        d = torch.load(os.path.join(dump, f))
        assert d["state"].shape[0] == 1, f"expected one prefill row, got {d['qlens']}"
        out[layer] = d
    for f in new:
        os.remove(os.path.join(dump, f))
    cached = (j.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    return out, cached, len(new)


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def pairs():
    out = []
    for f, picks in (("trajectories_short_production.json", SHORT), ("trajectories_long_production.json", LONG)):
        tr = {t["task"]: t["turns"] for t in json.load(open(os.path.join(D, f)))["trajectories"]}
        for s, k in picks:
            k = min(k, len(tr[s]) - 1)
            out.append((f.split("_")[1], s, k, tr[s][k - 1]["prompt_ids"], tr[s][k]["prompt_ids"]))
    return out


def main():
    dump, out, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    open(os.path.join(dump, "ENABLE"), "w").close()
    rows, stamp = [], int(time.time())
    try:
        for kind, s, k, prev, cur in pairs():
            salt = f"sc-{tag}-{stamp}-{kind}-{s}-{k}"
            c1, _, n1 = run(dump, cur, salt + "-c1")
            c2, _, _ = run(dump, cur, salt + "-c2")
            t = len(prev) - 32
            dr, _, _ = run(dump, cur[:t - 16] + cur[t:], salt + "-drop")
            run(dump, prev, salt + "-hot", n=64)
            h, hc, nh = run(dump, cur, salt + "-hot")
            row = {"kind": kind, "session": s, "turn": k, "len": len(cur), "prev_len": len(prev), "hot_cached": hc,
                   "cold_chunks": n1, "hot_chunks": nh}
            for layer in sorted(c1):
                ref = c1[layer]
                for name, x in (("cold2", c2), ("hot", h), ("drop16", dr)):
                    row[f"L{layer}_{name}_state_rel"] = rel(x[layer]["state"], ref["state"])
                    row[f"L{layer}_{name}_conv_maxabs"] = (x[layer]["conv"] - ref["conv"]).abs().max().item()
                row[f"L{layer}_hot_has_init"] = h[layer]["has_init"]
            print(json.dumps(row), flush=True)
            rows.append(row)
    finally:
        os.remove(os.path.join(dump, "ENABLE"))
    summ = {"tag": tag, "n": len(rows)}
    for key in rows[0]:
        if key.endswith("_rel") or key.endswith("_maxabs"):
            v = [r[key] for r in rows]
            summ[key] = {"mean": sum(v) / len(v), "max": max(v)}
    summ["hot_cached_mean"] = sum(r["hot_cached"] or 0 for r in rows) / len(rows)
    summ["prev_len_mean"] = sum(r["prev_len"] for r in rows) / len(rows)
    print("STATECHECK", json.dumps(summ), flush=True)
    json.dump({"summary": summ, "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
