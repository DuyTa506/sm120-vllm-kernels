"""VERIFIER (host python, no torch): dump prompt logprobs for prompts of several lengths, and greedy
continuations, from a running server; or compare two dumps.
  dump:    verify_logprobs.py dump <out.json>       (BENCH_URL env, default http://127.0.0.1:8127)
  compare: verify_logprobs.py compare <a.json> <b.json>
Prompts below 1024 tokens run entirely on the CUTLASS branch of the hybrid -> expected identical to the
cutlass server up to the server's own run-to-run noise; longer prompts take the b12x branch.
Each prompt is prefixed with a unique tag so prefix caching cannot reuse blocks across prompts."""
import glob, json, os, sys, urllib.request

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8127") + "/v1/completions"


def call(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


def text(chars, off):
    src = ""
    for p in sorted(glob.glob("/usr/lib/python3*/json/*.py")) + sorted(glob.glob("/usr/lib/python3*/http/*.py")):
        src += open(p, errors="ignore").read()
    return src[off:off + chars]


def dump(out):
    res = []
    for i, (chars, off) in enumerate([(2400, 0), (2700, 5000), (14000, 20000), (30000, 40000)]):
        prompt = f"[sample {i}]\n" + text(chars, off)
        j = call({"model": "bench", "prompt": prompt, "max_tokens": 64, "temperature": 0, "prompt_logprobs": 0})
        lps = [list(d.values())[0]["logprob"] for d in j["choices"][0]["prompt_logprobs"][1:] if d]
        res.append({"i": i, "tokens": len(lps) + 1, "lps": lps, "gen": j["choices"][0]["text"]})
        print(f"prompt {i}: {len(lps) + 1} tokens, nll {-sum(lps) / len(lps):.4f}", flush=True)
    json.dump(res, open(out, "w"))


def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    for x, y in zip(A, B):
        d = [abs(p - q) for p, q in zip(x["lps"], y["lps"])]
        n = len(d)
        same_gen = x["gen"] == y["gen"]
        common = next((k for k in range(min(len(x["gen"]), len(y["gen"]))) if x["gen"][k] != y["gen"][k]),
                      min(len(x["gen"]), len(y["gen"])))
        print(f"prompt {x['i']} ({x['tokens']} tok): nll {-sum(x['lps']) / n:.4f} vs {-sum(y['lps']) / n:.4f}; "
              f"|dlogprob| mean {sum(d) / n:.4f} max {max(d):.3f}; exactly equal {sum(v == 0 for v in d)}/{n}; "
              f"greedy 64-tok continuation identical {same_gen} (first diff at char {common})")


if __name__ == "__main__":
    dump(sys.argv[2]) if sys.argv[1] == "dump" else compare(sys.argv[2], sys.argv[3])
