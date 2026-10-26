"""Random short/long requests: finish mid-step (max_tokens 1..60), aborts (client closes the stream early),
long prompts. usage: stress.py <n_requests> <concurrency> [max_prompt_tokens]"""
import json, os, random, socket, sys, threading, time, urllib.request
URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8121")
N, C = int(sys.argv[1]), int(sys.argv[2])
MAXP = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
rng = random.Random(1)
words = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon").split()
ok = bad = aborted = 0
lock = threading.Lock()

def one(i):
    global ok, bad, aborted
    r = random.Random(i)
    prompt = " ".join(r.choice(words) for _ in range(r.randint(20, MAXP // 2)))
    body = {"model": "bench", "prompt": prompt, "max_tokens": r.randint(1, 60), "temperature": 0, "ignore_eos": True,
            "cache_salt": f"stress-{i}", "stream": False}
    try:
        if r.random() < 0.2:  # abort: stream and drop the connection after a few chunks
            body.update(stream=True, max_tokens=400)
            req = urllib.request.Request(URL + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            resp = urllib.request.urlopen(req, timeout=600)
            for _ in range(r.randint(1, 6)):
                resp.readline()
            resp.close()
            with lock: aborted += 1
        else:
            req = urllib.request.Request(URL + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            j = json.loads(urllib.request.urlopen(req, timeout=600).read())
            assert j["choices"][0]["finish_reason"] in ("length", "stop")
            with lock: ok += 1
    except Exception as e:
        with lock: bad += 1
        print("request", i, "failed:", repr(e)[:200])

idx = iter(range(N)); ilock = threading.Lock()
def worker():
    while True:
        with ilock:
            i = next(idx, None)
        if i is None: return
        one(i)
t0 = time.time()
ts = [threading.Thread(target=worker) for _ in range(C)]
[t.start() for t in ts]; [t.join() for t in ts]
print(f"stress: ok={ok} aborted={aborted} failed={bad} in {time.time()-t0:.0f}s")
