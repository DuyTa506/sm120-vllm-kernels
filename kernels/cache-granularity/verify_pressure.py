"""Verifier: eviction / memory-pressure / leak check for sub-block prefix-cache hits.

N sessions (distinct cache salts, ~P tokens each, needle inside the partial block before the hit position) are
warmed with concurrency C (64 decode tokens each), so the total exceeds the KV pool (3.98M tokens) and early
sessions get (partially) evicted. Then every session sends its next turn B = A + C + Q(X) (concurrency C, optional
think time) and must answer its own code. Afterwards the server idles and the KV usage gauge must return to ~0
(blocks held by a leaked CoW reference would show up here).

usage: BENCH_URL=... python3 verify_pressure.py <out.json> <tag> [--n 48] [--len 92000] [--conc 8] [--think 0]
"""
import argparse, json, re, statistics, threading, time, urllib.request

import verify_correctness as vc


def gauge(name):
    txt = urllib.request.urlopen(vc.URL + "/metrics", timeout=30).read().decode()
    vals = [float(m.group(1)) for m in re.finditer(rf"^{name}{{[^}}]*}} ([0-9.e+-]+)$", txt, re.M)]
    return sum(vals) if vals else None


def pool(jobs, conc):
    out, lock, q = [None] * len(jobs), threading.Lock(), list(range(len(jobs)))

    def w():
        while True:
            with lock:
                if not q:
                    return
                i = q.pop(0)
            try:
                out[i] = jobs[i]()
            except Exception as e:  # noqa: BLE001
                out[i] = {"error": repr(e)}
    ths = [threading.Thread(target=w) for _ in range(conc)]
    [t.start() for t in ths]; [t.join() for t in ths]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("tag")
    ap.add_argument("--n", type=int, default=48); ap.add_argument("--len", type=int, default=92000)
    ap.add_argument("--conc", type=int, default=8); ap.add_argument("--think", type=float, default=0.0)
    ap.add_argument("--order", default="fwd", help="fwd: query sessions oldest first; rev: newest first (partial eviction)")
    a = ap.parse_args()
    F, stamp = vc.Filler(), int(time.time())
    qX = vc.question("Kestrel")
    sess = []
    for i in range(a.n):
        A, cX, _, _ = vc.build_A(F, a.len - 2128 * (i % 3), "Kestrel")
        sess.append((A, cX, f"vp-{a.tag}-{stamp}-{i}", F.get(50000 + 37 * i, 64)))
    u0 = gauge("vllm:kv_cache_usage_perc")
    t0 = time.time()
    pool([lambda s=s: vc.warm(s[0], s[2]) for s in sess], a.conc)
    t_warm = time.time() - t0
    u1 = gauge("vllm:kv_cache_usage_perc")

    def nxt(s):
        if a.think:
            time.sleep(a.think)
        t = time.time()
        h = vc.gen(s[0] + s[3] + qX, s[2])
        return {"ok": vc.ok(h, s[1]), "cached": h["cached"], "len": len(s[0]) + 64 + len(qX),
                "T_A": vc.ckpt(len(s[0])), "text": h["text"], "ttft_s": round(time.time() - t, 3)}
    t0 = time.time()
    order = list(range(a.n)) if a.order == "fwd" else list(range(a.n))[::-1]
    res_o = pool([lambda s=sess[i]: nxt(s) for i in order], a.conc)
    res = [None] * a.n
    for j, i in enumerate(order):
        res[i] = res_o[j]
    t_hot = time.time() - t0
    errors = [r for r in res if "error" in r]
    good = [r for r in res if "error" not in r]
    time.sleep(8)
    u2 = gauge("vllm:kv_cache_usage_perc")
    summ = {"tag": a.tag, "n": a.n, "order": a.order, "prompt_tokens_total": sum(len(s[0]) for s in sess),
            "warm_s": round(t_warm, 1), "hot_s": round(t_hot, 1), "errors": len(errors),
            "hot_ok": sum(r["ok"] for r in good), "n_good": len(good),
            "full_hit_at_T": sum(r["cached"] == r["T_A"] for r in good),
            "partial_or_miss": sum(r["cached"] != r["T_A"] for r in good),
            "cached_mean": round(statistics.mean(r["cached"] for r in good), 1) if good else None,
            "kv_usage_before": u0, "kv_usage_after_warm": u1, "kv_usage_idle_after": u2,
            "wrong": [r for r in good if not r["ok"]][:10], "error_samples": errors[:3]}
    print("PRESSURE", json.dumps(summ), flush=True)
    json.dump({"summary": summ, "rows": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
