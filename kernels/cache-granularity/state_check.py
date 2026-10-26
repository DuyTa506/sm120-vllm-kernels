"""Engine-level hot-vs-cold check of the GDN recurrent state (needs the dump hook, see make_dump_build.py).

The end-to-end check (hot_cold.py) cannot resolve small state errors on this server: two identical cold runs already
differ by ~0.1 total variation in the first-token distribution (the server is not bit-deterministic). GDN layer 0 is
deterministic (its input is the token embedding; two identical cold runs give bit-identical layer-0 states), so its
state isolates exactly the property that matters: a prefix-cache hit at position T resumes from the state of the
first T tokens.

For each recorded (session, turn k) pair (prev = prompt of turn k-1, cur = prompt of turn k):
  warm    prev under salt H, decoding 64 tokens (as an agent turn; the request keeps running after its prompt-tail
          checkpoint was registered). All its prefill chunks are dumped; the chunk ending exactly at T (when the
          scheduler split there) gives the state that was stored for position T.
  hot     cur under salt H; cached_tokens = T; its first prefill chunk's initial state is the restored checkpoint.
  pre     cold run of cur[:T] (fresh salt): state of the first T tokens computed from scratch (other chunking).
  pre16   cold run of cur[:T-16]: what a checkpoint one 16-token unit too early would hold (error scale).
  cold    cold run of cur: final state, compared with the hot run's final state.
Checks (layer 0): restored == warm chunk-end state (bit-exact expected when that chunk exists);
rel L2(restored, pre) ~ chunking-order noise; rel L2(pre16, pre) = off-by-one-unit scale; rel L2(hot final, cold).

usage: BENCH_URL=... python3 state_check.py <dump_dir> <out.json> <tag>
"""
import json, os, sys, time, urllib.request

import torch

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8117")
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/agentic")
SHORT = [(s, k) for s in range(8) for k in (3, 9, 15)]
LONG = [(0, 20), (4, 60), (7, 50), (2, 100), (5, 110), (1, 40)]


def post(body):
    r = urllib.request.Request(URL + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())


def run(dump, prompt, salt, n=1):
    """Send one request; return its layer-0/2 dumps in order as {layer: [(chunk_start, chunk_end, dict)]}."""
    before = set(os.listdir(dump))
    j = post({"model": "bench", "prompt": prompt, "max_tokens": n, "min_tokens": n, "ignore_eos": True,
              "temperature": 0, "cache_salt": salt})
    time.sleep(0.2)
    new = sorted(set(os.listdir(dump)) - before - {"ENABLE"})
    cached = (j.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    out, pos = {}, {}
    for f in new:  # <counter>_<layer>.pt in execution order
        layer = f.split("_")[1][:-3]
        d = torch.load(os.path.join(dump, f))
        os.remove(os.path.join(dump, f))
        assert len(d["qlens"]) == 1, f"expected one prefill row per step, got {d['qlens']}"
        start = pos.get(layer, cached)
        pos[layer] = start + d["qlens"][0]
        out.setdefault(layer, []).append((start, pos[layer], d))
    return out, cached


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
            warm, _ = run(dump, prev, salt + "-hot", n=64)
            hot, T = run(dump, cur, salt + "-hot")
            row = {"kind": kind, "session": s, "turn": k, "len": len(cur), "prev_len": len(prev), "hit": T,
                   "hot_chunks": [(a, b) for a, b, _ in hot["0"]], "warm_chunk_ends": [b for _, b, _ in warm["0"]]}
            if T == 0:
                rows.append(row); print(json.dumps(row), flush=True); continue
            pre, _ = run(dump, cur[:T], salt + "-pre")
            pre16, _ = run(dump, cur[:T - 16], salt + "-pre16")
            cold, _ = run(dump, cur, salt + "-cold")
            for layer in ("0", "2"):
                restored = hot[layer][0][2]["init"]
                row[f"L{layer}_hot_has_init"] = hot[layer][0][2]["has_init"][0]
                ref = pre[layer][-1][2]["state"]
                ends = {b: d["state"] for _, b, d in warm[layer]}
                if T in ends:
                    row[f"L{layer}_restored_vs_warm_maxabs"] = (restored - ends[T]).abs().max().item()
                row[f"L{layer}_restored_vs_pre_rel"] = rel(restored, ref)
                row[f"L{layer}_pre16_vs_pre_rel"] = rel(pre16[layer][-1][2]["state"], ref)
                row[f"L{layer}_hot_final_vs_cold_rel"] = rel(hot[layer][-1][2]["state"], cold[layer][-1][2]["state"])
            print(json.dumps(row), flush=True)
            rows.append(row)
    finally:
        os.remove(os.path.join(dump, "ENABLE"))
    summ = {"tag": tag, "n": len(rows), "hit_mean": sum(r["hit"] for r in rows) / len(rows),
            "prev_len_mean": sum(r["prev_len"] for r in rows) / len(rows)}
    keys = sorted({k for r in rows for k in r if k.startswith("L") and not k.endswith("has_init")})
    for key in keys:
        v = [r[key] for r in rows if key in r]
        summ[key] = {"n": len(v), "mean": sum(v) / len(v), "min": min(v), "max": max(v)}
    summ["has_init_all"] = all(r.get("L0_hot_has_init", True) for r in rows)
    print("STATECHECK", json.dumps(summ), flush=True)
    json.dump({"summary": summ, "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
