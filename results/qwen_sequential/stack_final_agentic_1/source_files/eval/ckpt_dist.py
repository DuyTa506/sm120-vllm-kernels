"""First-generated-token distribution after a checkpoint hit (hot) vs cold, many prompts.
Reports mean total-variation distance over top-20 and top-1 agreement. usage: ckpt_dist.py <tag>"""
import glob, json, os, math, sys, urllib.request
URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8100"); tag = sys.argv[1]
def post(body):
    r = urllib.request.Request(URL + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=600).read())
def hits():
    m = urllib.request.urlopen(URL + "/metrics", timeout=30).read().decode()
    return sum(float(l.split()[-1]) for l in m.splitlines() if l.startswith("vllm:prefix_cache_hits_total"))
files = sorted(glob.glob("/usr/lib/python3*/*.py"), key=lambda p: -len(open(p, errors="ignore").read()))
corpus = "\n\n".join(open(p, errors="ignore").read() for p in files[:60])
qs = ["What does the last function above return?", "Name one class defined above.", "Which module is imported first?",
      "Summarize the code above in one sentence.", "What is the purpose of the code above?", "List one constant defined above."]
tv_all, top1 = [], 0; hit_tot = 0; N = 12
for i in range(N):
    start = 150000 * i % (len(corpus) - 80000)
    a = f"<|im_start|>user\n[{tag}-{i}]\n" + corpus[start:start + 40000]
    b = a + corpus[start + 40000:start + 44000] + f"\n\n{qs[i % len(qs)]}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    g = {"model": "bench", "max_tokens": 1, "temperature": 0, "logprobs": 20}
    cold = post({**g, "prompt": b, "cache_salt": f"c-{tag}-{i}"})["choices"][0]["logprobs"]["top_logprobs"][0]
    post({"model": "bench", "prompt": a, "max_tokens": 1, "temperature": 0, "cache_salt": f"h-{tag}-{i}"})
    h1 = hits(); hot = post({**g, "prompt": b, "cache_salt": f"h-{tag}-{i}"})["choices"][0]["logprobs"]["top_logprobs"][0]; hit_tot += hits() - h1
    keys = set(cold) | set(hot)
    pc = {k: math.exp(cold.get(k, -30)) for k in keys}; ph = {k: math.exp(hot.get(k, -30)) for k in keys}
    tv = 0.5 * sum(abs(pc[k] - ph[k]) for k in keys); tv_all.append(tv)
    top1 += max(cold, key=cold.get) == max(hot, key=hot.get)
print(f"{tag}: hits/prompt={hit_tot/N:.0f}  mean TV(top20)={sum(tv_all)/N:.4f}  max TV={max(tv_all):.4f}  top1 agree={top1}/{N}")
print("  per-prompt TV:", " ".join(f"{x:.3f}" for x in tv_all))
