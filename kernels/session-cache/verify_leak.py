"""Verifier: block-leak / TTL check. Reads vllm:kv_cache_usage_perc while sessions are retained, waits past the
session TTL, pokes the scheduler (maintain() only runs inside schedule()), and reads the usage again. After TTL
eviction and with no running request, usage must return to ~0 (every retained block freed).

usage: verify_leak.py <wait_s> <out.json>
"""
import json, sys, time, urllib.request
import session_bench as sb


def usage():
    m = urllib.request.urlopen(sb.URL + "/metrics", timeout=30).read().decode()
    vals = [float(l.split()[-1]) for l in m.splitlines() if l.startswith("vllm:kv_cache_usage_perc")]
    return vals[0] if vals else None


def poke(i, sid=None):
    body = {"model": "bench", "prompt": [100 + i, 200, 300, 400], "max_tokens": 2, "temperature": 0,
            "cache_salt": f"poke-{time.time()}-{i}"}
    if sid:
        body["session_id"] = sid
    r = urllib.request.Request(sb.URL + "/v1/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    urllib.request.urlopen(r, timeout=60).read()


wait, out = float(sys.argv[1]), sys.argv[2]
res = {}
poke(0); time.sleep(2); poke(1); time.sleep(3)
res["usage_retained"] = usage()
print("usage with sessions retained:", res["usage_retained"], flush=True)
time.sleep(wait)
poke(2, sid=f"leakprobe-{time.time()}")  # schedule() -> maintain() evicts by TTL; on_finish logs the counts
time.sleep(2); poke(3); time.sleep(2); poke(4); time.sleep(5)
res["usage_after_ttl"] = usage()
print("usage after TTL eviction:", res["usage_after_ttl"], flush=True)
json.dump(res, open(out, "w"))
