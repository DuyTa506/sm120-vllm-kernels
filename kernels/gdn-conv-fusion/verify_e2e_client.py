"""VERIFIER host-side client (stdlib only) for verify_e2e.sh.

1. Logit check: the same random-token prompt is prefilled with the fused path off, off,
   on, on (distinct cache_salt so there are no prefix-cache hits) and prompt_logprobs of
   all prompt tokens are compared. If off/off is deterministic, on/off should be too.
2. Interleaved TTFT: max_tokens=1 requests of 8k and 16k random tokens (unique per
   request), fused toggled per request in ABBA order; median request latency per arm.
"""

import json
import os
import random
import statistics
import sys
import time
import urllib.request

URL, FLAGDIR = sys.argv[1], sys.argv[2]
FLAG = os.path.join(FLAGDIR, "fused_on")


def set_flag(on):
    if on:
        open(FLAG, "w").close()
    elif os.path.exists(FLAG):
        os.remove(FLAG)


def post(body):
    r = urllib.request.Request(URL + "/v1/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    out = json.loads(urllib.request.urlopen(r, timeout=600).read())
    return out, (time.perf_counter() - t0) * 1e3


def rand_prompt(n, seed):
    rng = random.Random(seed)
    return [rng.randrange(1000, 150000) for _ in range(n)]


def prompt_lp(ids, on, salt):
    set_flag(on)
    out, _ = post({"model": "bench", "prompt": ids, "max_tokens": 4, "temperature": 0,
                   "prompt_logprobs": 0, "logprobs": 1, "cache_salt": salt})
    ch = out["choices"][0]
    plp = ch.get("prompt_logprobs") or []
    vals = []
    for e in plp[1:]:
        if e is None:
            vals.append(None)
            continue
        # one entry per position: {token_id: {"logprob": ...}}
        v = list(e.values())[0]
        vals.append(v["logprob"] if isinstance(v, dict) else v)
    return vals, ch["text"]


def maxdiff(a, b):
    return max(abs(x - y) for x, y in zip(a, b) if x is not None and y is not None)


# warm both arms (first fused call JIT-compiles)
for on in (False, True, False, True):
    set_flag(on)
    post({"model": "bench", "prompt": rand_prompt(8192, 10_000 + on), "max_tokens": 1})
    post({"model": "bench", "prompt": rand_prompt(300, 20_000 + on), "max_tokens": 1})

print("== logit check (prompt_logprobs over all prompt positions)")
for n, seed in [(8192, 1), (3000, 2), (16384, 3)]:
    ids = rand_prompt(n, seed)
    r = {}
    for tag, on in [("off1", False), ("off2", False), ("on1", True), ("on2", True)]:
        r[tag] = prompt_lp(ids, on, f"v{seed}-{tag}-{time.time()}")
    lp = {k: v[0] for k, v in r.items()}
    print(f"  len {n}: off/off maxdiff={maxdiff(lp['off1'], lp['off2']):.3e} "
          f"on/on={maxdiff(lp['on1'], lp['on2']):.3e} off/on={maxdiff(lp['off1'], lp['on1']):.3e} "
          f"| text off1={r['off1'][1]!r} on1={r['on1'][1]!r}")

print("== interleaved latency (max_tokens=1, unique random prompts, ABBA order)")
seed = 100_000
for n, reps in [(8192, 40), (16384, 24)]:
    t = {False: [], True: []}
    order = [False, True, True, False] * (reps // 2)
    for on in order:
        seed += 1
        set_flag(on)
        _, ms = post({"model": "bench", "prompt": rand_prompt(n, seed), "max_tokens": 1})
        t[on].append(ms)
    mo, mf = statistics.median(t[False]), statistics.median(t[True])
    # paired differences of adjacent (off, on) requests
    print(f"  {n:6d} tok: off median {mo:.1f} ms (n={len(t[False])}, iqr "
          f"{statistics.quantiles(t[False], n=4)[0]:.1f}-{statistics.quantiles(t[False], n=4)[2]:.1f}) | "
          f"on median {mf:.1f} ms (n={len(t[True])}, iqr {statistics.quantiles(t[True], n=4)[0]:.1f}-"
          f"{statistics.quantiles(t[True], n=4)[2]:.1f}) | saved {mo - mf:.1f} ms")
set_flag(False)
