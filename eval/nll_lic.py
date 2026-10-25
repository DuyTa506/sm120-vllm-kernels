"""licenses-only repeated NLL (unique prefixes). usage: nll_lic.py out.json reps"""
import glob, json, os, statistics, sys, urllib.request
reps = int(sys.argv[2]); paths = sorted(glob.glob("/usr/share/common-licenses/*"))
full = "\n\n".join(open(p, errors="ignore").read() for p in paths)
def nll(t):
    body = {"model": "bench", "prompt": t, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}
    j = json.loads(urllib.request.urlopen(urllib.request.Request(os.environ.get("BENCH_URL", "http://127.0.0.1:8100") + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600).read())
    lps = [list(d.values())[0]["logprob"] for d in j["choices"][0]["prompt_logprobs"][8:] if d]
    return -sum(lps) / len(lps)
out = {}
for chars in (30000, 60000):
    vals = [nll(f"[lic {chars}-{r} q{r*7919}z]\n" + full[:chars]) for r in range(reps)]
    out[chars] = vals
    print(f"licenses {chars} mean={statistics.mean(vals):.4f} sd={statistics.pstdev(vals):.4f} se={statistics.pstdev(vals)/len(vals)**.5:.4f}", flush=True)
json.dump(out, open(sys.argv[1], "w"))
