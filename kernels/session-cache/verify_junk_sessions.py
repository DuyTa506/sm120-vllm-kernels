"""Verifier: create N idle sessions (short distinct prompts, 2 output tokens, unique session_id each). Each retained
entry pins ~4 blocks (1 attention tail + 3 Mamba snapshot blocks) until TTL / LRU / <2 % free.
usage: verify_junk_sessions.py N [conc] [nosid]  (nosid: same requests without session_id = control, nothing pinned)"""
import json, sys, threading, time, urllib.request
import session_bench as sb

N = int(sys.argv[1]); C = int(sys.argv[2]) if len(sys.argv) > 2 else 16
NOSID = len(sys.argv) > 3 and sys.argv[3] == "nosid"
tag = f"junk-{int(time.time())}"
todo = list(range(N)); lock = threading.Lock()


def worker():
    while True:
        with lock:
            if not todo:
                return
            i = todo.pop()
        prompt = [1000 + (i * 7 + j) % 50000 for j in range(120)]
        body = {"model": "bench", "prompt": prompt, "max_tokens": 2, "temperature": 0,
                "cache_salt": f"{tag}-{i}"}
        if not NOSID:
            body["session_id"] = f"{tag}-{i}"
        r = urllib.request.Request(sb.URL + "/v1/completions", json.dumps(body).encode(),
                                   {"Content-Type": "application/json"})
        urllib.request.urlopen(r, timeout=120).read()


t0 = time.time()
th = [threading.Thread(target=worker) for _ in range(C)]
[x.start() for x in th]; [x.join() for x in th]
m = urllib.request.urlopen(sb.URL + "/metrics", timeout=30).read().decode()
u = [l for l in m.splitlines() if l.startswith("vllm:kv_cache_usage_perc")]
print(f"created {N} idle sessions in {time.time() - t0:.1f}s; {u}", flush=True)
