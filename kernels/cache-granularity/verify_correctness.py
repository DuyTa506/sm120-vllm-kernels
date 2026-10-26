"""Verifier: needle-recall hot-vs-cold checks for sub-block prefix-cache hits (--prefix-match-unit).

The first-token TV test (hot_cold.py) cannot resolve state errors on this non-deterministic server. This test uses
prompts whose answer depends on content stored exactly where a fine-grained hit is served from: a random access code
("needle") placed inside the partial attention block [last block boundary B0, hit position T), and questions whose
answer changes if the server reuses the wrong prefix. A cold run (fresh cache salt) must answer correctly for a case
to count; then the hot run must answer correctly too. Hot-vs-cold first-token TV and greedy agreement are recorded.

Cases per prompt length (each with its own cache salt; A is the "previous turn", warmed with 64 decode tokens):
  ext        B = A + C + Q(X)                    plain agentic extension, hit at T_A
  tiny       A ends with the question; B = A + first 3 answer tokens (extension shorter than one hash unit)
  swap       A' = A with X's code replaced (same length, divergence INSIDE the partial block before T_A); warm A,
             then B = A' + C + Q(X) must answer the NEW code (a wrong hash-chain hit would answer A's code)
  div_in     B = A[:T_A-5] + N(Y,cB) + Q(Y)      divergence inside the last hash unit before T_A
  div_at     B = A[:T_A]   + N(Y,cB) + Q(Y)      divergence exactly at T_A
  div_after  B = A[:P_A-2] + N(Y,cB) + Q(Y)      divergence after the attention partial boundary (hit at T_A)
  trunc      B = A[:end(NX)+40] + Q(X)           shorter than A, cut inside the partial block
Concurrency (section "conc"):
  live       A decodes 400 tokens; 4 consumers B_i = A + C_i + Q(X) are sent while A is still decoding
  race       A and B = A + C + Q(X) sent at the same time (B arrives while A is prefilling)
  siblings   8 prompts share a 20k prefix + distinct needles, sent concurrently under ONE salt, then 8 hot
             extensions concurrently

usage: BENCH_URL=... python3 verify_correctness.py <out.json> <tag> [--lengths 9800,31000,75000] [--sections cases,conc]
"""
import argparse, json, math, os, random, threading, time, urllib.request

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8121")
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results/agentic")
BLOCK, UNIT = 2128, 16


def post(path, body, timeout=900):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=timeout).read())


def tok(text):
    return post("/tokenize", {"model": "bench", "prompt": text, "add_special_tokens": False})["tokens"]


def gen(prompt, salt, n=24, logprobs=5):
    j = post("/v1/completions", {"model": "bench", "prompt": prompt, "max_tokens": n, "min_tokens": n,
                                 "temperature": 0, "logprobs": logprobs, "cache_salt": salt})
    ch = j["choices"][0]
    cached = (j.get("usage", {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return {"text": ch["text"], "toks": ch["logprobs"]["tokens"], "top": ch["logprobs"]["top_logprobs"][0],
            "cached": cached}


def warm(prompt, salt, n=64):
    post("/v1/completions", {"model": "bench", "prompt": prompt, "max_tokens": n, "min_tokens": n,
                             "ignore_eos": True, "temperature": 0, "cache_salt": salt})


def tv(a, b):
    keys = set(a) | set(b)
    return 0.5 * sum(abs(math.exp(a.get(k, -30)) - math.exp(b.get(k, -30))) for k in keys)


def agree(x, y):
    n = 0
    while n < min(len(x), len(y)) and x[n] == y[n]:
        n += 1
    return n


RNG = random.Random(1234)


def code():
    return "-".join(str(RNG.randrange(1000, 10000)) for _ in range(3))


def needle(name, c):
    return tok(f"\n\n[NOTE] The access code for vault {name} is {c}. Remember it exactly.\n\n")


def question(name):
    # chat-formatted (Qwen3 template, empty think block) so the model answers instead of emitting EOS
    return tok(f"<|im_end|>\n<|im_start|>user\nWhat is the access code for vault {name}? Reply with the code "
               f"only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nThe access code for vault {name} is")


def ok(res, c):
    return c in res["text"].replace(" ", "")


class Filler:
    def __init__(self):
        tr = json.load(open(os.path.join(D, "trajectories_long_production.json")))["trajectories"]
        self.toks = max((t["turns"][-1]["prompt_ids"] for t in tr), key=len)  # ~126k recorded tokens

    def get(self, start, n):
        return self.toks[start:start + n]


def ckpt(p):  # MTP: hit position for a prompt of length p
    return max(p // UNIT * UNIT - UNIT, 0)


def build_A(F, target, name):
    """A = F[:a] + N(name, cX) + F[...] with P_A ~ target, P_A % 16 == 8, needle inside [B0, T_A - 150)."""
    cX = code()
    nx = needle(name, cX)
    P = target - target % BLOCK + 1100 + 8 - (1100 % UNIT)  # P mod 2128 ~ 1100, P mod 16 == 8
    T = ckpt(P); B0 = T // BLOCK * BLOCK
    nx_start = B0 + (T - B0) // 2 - len(nx) // 2  # middle of the partial block region
    assert B0 < nx_start and nx_start + len(nx) < T - 150
    A = F.get(0, nx_start) + nx
    A = A + F.get(nx_start + 5000, P - len(A))
    assert len(A) == P, (len(A), P)
    return A, cX, nx_start, nx_start + len(nx)


def run_case(name, B, expect, salt, warm_prompt=None, cold2=False):
    cold = gen(B, salt + "-cold")
    row = {"case": name, "len": len(B), "expect": expect, "cold_ok": ok(cold, expect), "cold_text": cold["text"]}
    if cold2:
        c2 = gen(B, salt + "-cold2")
        row["cold2_tv"] = round(tv(c2["top"], cold["top"]), 4); row["cold2_agree"] = agree(c2["toks"], cold["toks"])
        row["cold2_ok"] = ok(c2, expect)
    if warm_prompt is not None:
        warm(warm_prompt, salt + "-hot")
    hot = gen(B, salt + "-hot")
    row.update({"hot_ok": ok(hot, expect), "hot_text": hot["text"], "hot_cached": hot["cached"],
                "tv": round(tv(hot["top"], cold["top"]), 4), "agree": agree(hot["toks"], cold["toks"])})
    print(json.dumps(row), flush=True)
    return row


def cases(F, L, stamp, tag):
    rows = []
    s = f"vc-{tag}-{stamp}-{L}"
    A, cX, ns, ne = build_A(F, L, "Kestrel")
    P = len(A); T = ckpt(P)
    base = {"P_A": P, "T_A": T, "B0": T // BLOCK * BLOCK, "needle": [ns, ne]}
    print("A", json.dumps(base), flush=True)
    C = F.get(90000 % max(1, len(F.toks) - 200), 64)  # stands in for the previous completion + tool result
    qX = question("Kestrel")
    for r in [
        run_case("ext", A + C + qX, cX, s + "-ext", A, cold2=True),
        run_case("tiny", A + qX + tok(" " + cX[:3]), cX[3:], s + "-tiny", A + qX),
        run_case("trunc", A[:ne + 40] + qX, cX, s + "-trunc", A),
    ]:
        rows.append(dict(r, **base))
    # swap: same length, X's code replaced inside the partial block; warm with the original A
    c2 = code()
    nx2 = needle("Kestrel", c2)
    if len(nx2) == ne - ns:
        A2 = A[:ns] + nx2 + A[ne:]
        rows.append(dict(run_case("swap", A2 + C + qX, c2, s + "-swap", A), stale=cX, **base))
    cB = code()
    nY, qY = needle("Osprey", cB), question("Osprey")
    for name, cut in (("div_in", T - 5), ("div_at", T), ("div_after", P - 2)):
        rows.append(dict(run_case(name, A[:cut] + nY + qY, cB, s + "-" + name, A), **base))
    return rows


def threaded(fns):
    out = [None] * len(fns)

    def wrap(i, f):
        try:
            out[i] = f()
        except Exception as e:  # noqa: BLE001
            out[i] = {"error": repr(e)}
    ths = [threading.Thread(target=wrap, args=(i, f)) for i, f in enumerate(fns)]
    [t.start() for t in ths]; [t.join() for t in ths]
    return out


def conc(F, stamp, tag):
    rows = []
    s = f"vcc-{tag}-{stamp}"
    qX = question("Kestrel")
    # live: consumers while the producer decodes
    A, cX, _, _ = build_A(F, 40000, "Kestrel")
    started = threading.Event()

    def producer():
        body = {"model": "bench", "prompt": A, "max_tokens": 400, "min_tokens": 400, "ignore_eos": True,
                "temperature": 0, "cache_salt": s + "-live", "stream": True}
        r = urllib.request.Request(URL + "/v1/completions", json.dumps(body).encode(),
                                   {"Content-Type": "application/json"})
        n = 0
        with urllib.request.urlopen(r, timeout=900) as resp:
            for line in resp:
                if line.startswith(b"data: {"):
                    n += 1
                    if n == 3:
                        started.set()
        return {"producer_chunks": n}

    def consumer(i):
        started.wait(600)
        time.sleep(0.05 * i)
        B = A + F.get(70000 + 300 * i, 64) + qX
        h = gen(B, s + "-live")
        return {"case": "live", "i": i, "hot_ok": ok(h, cX), "hot_cached": h["cached"], "hot_text": h["text"]}
    out = threaded([producer] + [lambda i=i: consumer(i) for i in range(4)])
    cold = gen(A + F.get(70000, 64) + qX, s + "-live-cold")
    for r in out[1:]:
        r["cold_ok"] = ok(cold, cX); r["len"] = len(A) + 64 + len(qX); print(json.dumps(r), flush=True); rows.append(r)
    print(json.dumps(out[0]), flush=True)
    # race: producer and consumer at the same time
    A, cX, _, _ = build_A(F, 30000, "Kestrel")
    B = A + F.get(71000, 64) + qX
    out = threaded([lambda: warm(A, s + "-race"), lambda: gen(B, s + "-race")])
    h = out[1]; cold = gen(B, s + "-race-cold")
    r = {"case": "race", "len": len(B), "hot_ok": ok(h, cX), "cold_ok": ok(cold, cX), "hot_cached": h.get("cached"),
         "hot_text": h.get("text"), "tv": round(tv(h["top"], cold["top"]), 4)}
    print(json.dumps(r), flush=True); rows.append(r)
    # siblings: shared 20k prefix, distinct needles, one salt, concurrent producers then concurrent consumers
    shared = F.get(0, 20000)
    sib = []
    for i in range(8):
        c = code()
        body = shared + needle("Heron", c) + F.get(30000 + 2000 * i, 1500 + 97 * i)
        sib.append((body, c, F.get(60000 + 500 * i, 64)))
    out = threaded([lambda b=b: warm(b, s + "-sib") for b, _, _ in sib])
    qH = question("Heron")
    out = threaded([lambda b=b, cc=cc: gen(b + cc + qH, s + "-sib") for b, _, cc in sib])
    for i, ((b, c, cc), h) in enumerate(zip(sib, out)):
        cold = gen(b + cc + qH, s + f"-sib-cold{i}")
        r = {"case": "siblings", "i": i, "len": len(b) + 64 + len(qH), "hot_ok": ok(h, c), "cold_ok": ok(cold, c),
             "hot_cached": h.get("cached"), "hot_text": h.get("text"), "tv": round(tv(h["top"], cold["top"]), 4)}
        print(json.dumps(r), flush=True); rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("tag")
    ap.add_argument("--lengths", default="9800,31000,75000")
    ap.add_argument("--sections", default="cases,conc")
    a = ap.parse_args()
    F, stamp, rows = Filler(), int(time.time()), []
    if "cases" in a.sections:
        for L in [int(x) for x in a.lengths.split(",")]:
            rows += cases(F, L, stamp, a.tag)
    if "conc" in a.sections:
        rows += conc(F, stamp, a.tag)
    valid = [r for r in rows if r.get("cold_ok")]
    summ = {"tag": a.tag, "n": len(rows), "cold_ok": len(valid),
            "hot_ok_where_cold_ok": sum(r.get("hot_ok", False) for r in valid),
            "hot_ok_all": sum(r.get("hot_ok", False) for r in rows),
            "failures": [{k: r.get(k) for k in ("case", "len", "expect", "hot_text", "cold_text", "hot_cached")}
                         for r in rows if r.get("cold_ok") and not r.get("hot_ok")],
            "tv_mean": round(sum(r["tv"] for r in rows if "tv" in r) / max(1, sum("tv" in r for r in rows)), 4),
            "cold2_tv_mean": [r["cold2_tv"] for r in rows if "cold2_tv" in r]}
    print("VERIFY", json.dumps(summ), flush=True)
    json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
