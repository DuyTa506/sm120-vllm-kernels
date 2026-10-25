"""Hot-vs-cold correctness on recorded agentic turns.

For each (session, turn k) pair:
  cold1, cold2  turn k with fresh cache salts (cold2 - cold1 = the server's own run-to-run noise: this model/server
                is not bit-deterministic, so identical cold runs already differ)
  hot           warm the cache with turn k-1 (decoding 64 tokens after the prompt, as an agent turn would, so the
                running request keeps mutating its state after the prompt-tail checkpoint was registered), then turn
                k under the same salt (hit lands near the end of turn k-1)
  dup16, drop16 sensitivity controls, cold: turn k with 16 tokens just before the previous prompt's end duplicated /
                removed. A Mamba state that is off by one 16-token hash unit would look like this to the GDN layers.
Compares greedy continuations (32 tokens) and the first-token top-5 distribution (total variation) against cold1.

usage: BENCH_URL=... python3 hot_cold.py <out.json> <tag>
"""
import json, math, os, sys, time, urllib.request

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8117")
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/agentic")
N_GEN = 32
SHORT = [(s, k) for s in range(8) for k in (3, 7, 11, 15)]
LONG = [(0, 20), (4, 60), (7, 50), (2, 100), (5, 110), (1, 40)]


def post(body):
    r = urllib.request.Request(URL + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=900).read())


def gen(prompt, salt, n=N_GEN):
    j = post({"model": "bench", "prompt": prompt, "max_tokens": n, "temperature": 0, "logprobs": 5,
              "cache_salt": salt})
    ch = j["choices"][0]
    cached = (j.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    return ch["logprobs"]["tokens"], ch["logprobs"]["top_logprobs"][0], cached


def tv(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(math.exp(a.get(k, -30)) - math.exp(b.get(k, -30))) for k in keys)


def agree(x, y):
    n = 0
    while n < min(len(x), len(y)) and x[n] == y[n]:
        n += 1
    return n


def pairs():
    out = []
    for f, picks in (("trajectories_short_production.json", SHORT), ("trajectories_long_production.json", LONG)):
        tr = {t["task"]: t["turns"] for t in json.load(open(os.path.join(D, f)))["trajectories"]}
        for s, k in picks:
            k = min(k, len(tr[s]) - 1)
            out.append((f.split("_")[1], s, k, tr[s][k - 1]["prompt_ids"], tr[s][k]["prompt_ids"]))
    return out


def main():
    out, tag = sys.argv[1], sys.argv[2]
    rows, stamp = [], int(time.time())
    for kind, s, k, prev, cur in pairs():
        salt = f"hc-{tag}-{stamp}-{kind}-{s}-{k}"
        c1, l1, cc1 = gen(cur, salt + "-cold1")
        c2, l2, _ = gen(cur, salt + "-cold2")
        t = len(prev) - 32
        d, ld, _ = gen(cur[:t] + cur[t - 16:t] + cur[t:], salt + "-dup")
        r, lr, _ = gen(cur[:t - 16] + cur[t:], salt + "-drop")
        post({"model": "bench", "prompt": prev, "max_tokens": 64, "min_tokens": 64, "ignore_eos": True,
              "temperature": 0, "cache_salt": salt + "-hot"})
        h, lh, ch = gen(cur, salt + "-hot")
        row = {"kind": kind, "session": s, "turn": k, "prev_len": len(prev), "len": len(cur),
               "cold_cached": cc1, "hot_cached": ch}
        for name, toks, lp in (("cold2", c2, l2), ("hot", h, lh), ("dup16", d, ld), ("drop16", r, lr)):
            row[name] = {"agree": agree(toks, c1), "tv": round(tv(lp, l1), 4),
                         "top1": max(lp, key=lp.get) == max(l1, key=l1.get)}
        print(json.dumps(row), flush=True)
        rows.append(row)
    n = len(rows)
    summ = {"tag": tag, "n": n, "hot_cached_mean": round(sum(r["hot_cached"] or 0 for r in rows) / n, 1),
            "prev_len_mean": round(sum(r["prev_len"] for r in rows) / n, 1)}
    for name in ("cold2", "hot", "dup16", "drop16"):
        tvs = sorted(r[name]["tv"] for r in rows)
        summ[name] = {"tv_mean": round(sum(tvs) / n, 4), "tv_median": tvs[n // 2], "tv_max": tvs[-1],
                      "top1_same": sum(r[name]["top1"] for r in rows),
                      "greedy32_same": sum(r[name]["agree"] == N_GEN for r in rows),
                      "agree_mean": round(sum(r[name]["agree"] for r in rows) / n, 1)}
    print("HOTCOLD", json.dumps(summ), flush=True)
    json.dump({"summary": summ, "rows": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
