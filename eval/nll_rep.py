"""Prompt NLL with unique per-request prefixes (no prefix-cache reuse), repeated. usage: nll_rep.py out.json [reps]"""
import glob, json, os, statistics, sys, urllib.request
reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
docs = {
    "licenses": sorted(glob.glob("/usr/share/common-licenses/*")),
    "py_asyncio": sorted(glob.glob("/usr/lib/python3*/asyncio/*.py")),
    "py_email": sorted(glob.glob("/usr/lib/python3*/email/*.py")),
}
def text_of(paths, n):
    return "\n\n".join(open(p, errors="ignore").read() for p in paths)[:n]
def nll(t):
    body = {"model": "bench", "prompt": t, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}
    j = json.loads(urllib.request.urlopen(urllib.request.Request(os.environ.get("BENCH_URL", "http://127.0.0.1:8100") + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600).read())
    lps = [list(d.values())[0]["logprob"] for d in j["choices"][0]["prompt_logprobs"][8:] if d]  # skip the unique prefix
    return len(lps), -sum(lps) / len(lps)
out = {}
for name, paths in docs.items():
    for chars in (30000, 60000):
        vals = []
        for r in range(reps):
            n, v = nll(f"[sample {name}-{chars}-{r} 7f3a{r}9]\n" + text_of(paths, chars))
            vals.append(v)
        out[f"{name}-{chars}"] = vals
        print(f"{name:10s} {chars:6d} tok~{n:6d}  mean={statistics.mean(vals):.4f}  sd={statistics.pstdev(vals):.4f}  {['%.4f' % x for x in vals]}", flush=True)
json.dump(out, open(sys.argv[1], "w"))
