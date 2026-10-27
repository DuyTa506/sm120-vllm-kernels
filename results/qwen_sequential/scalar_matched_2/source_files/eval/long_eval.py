"""Long-context quality probe against the bench server (127.0.0.1:8100):
1) prompt NLL on long real documents (prompt_logprobs), 2) needle-in-a-haystack retrieval.
usage: long_eval.py <out.json>"""
import glob, json, os, random, sys, urllib.request

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8100") + "/v1/completions"


def call(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


def doc_text(paths, max_chars):
    out = []
    for p in paths:
        try:
            out.append(open(p, errors="ignore").read())
        except OSError:
            pass
        if sum(map(len, out)) > max_chars:
            break
    return "\n\n".join(out)[:max_chars]


res = {"nll": [], "needle": []}

# 1) NLL on long documents (licenses + python stdlib sources/docs: real, varied text)
docs = {
    "licenses": sorted(glob.glob("/usr/share/common-licenses/*")),
    "py_stdlib": sorted(glob.glob("/usr/lib/python3*/asyncio/*.py"))[:40],
    "py_email": sorted(glob.glob("/usr/lib/python3*/email/*.py"))[:40],
}
for name, paths in docs.items():
    for chars in (30000, 60000):
        text = doc_text(paths, chars)
        j = call({"model": "bench", "prompt": text, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0})
        lps = [list(d.values())[0]["logprob"] for d in j["choices"][0]["prompt_logprobs"][1:] if d]
        n = len(lps)
        res["nll"].append({"doc": name, "tokens": n, "nll": -sum(lps) / n,
                           "nll_last_half": -sum(lps[n // 2:]) / (n - n // 2)})
        print(f"NLL {name:10s} tokens={n:6d} mean={res['nll'][-1]['nll']:.4f} last_half={res['nll'][-1]['nll_last_half']:.4f}", flush=True)

# 2) needle in a haystack
filler_src = doc_text(docs["licenses"] + docs["py_email"], 400000)
rng = random.Random(1234)
for target_chars in (60000, 120000):  # ~16k and ~32k tokens
    for depth in (0.05, 0.25, 0.5, 0.75, 0.95):
        for rep in range(2):
            code = f"{rng.randint(100000, 999999)}"
            city = rng.choice(["Hanoi", "Lima", "Oslo", "Nairobi", "Hue", "Quito", "Perth", "Riga"])
            needle = f"\nIMPORTANT NOTE: The secret access code for the {city} office is {code}.\n"
            start = rng.randint(0, len(filler_src) - target_chars - 1)
            hay = filler_src[start:start + target_chars]
            pos = int(len(hay) * depth)
            ctx = hay[:pos] + needle + hay[pos:]
            prompt = ("<|im_start|>user\n" + ctx + f"\n\nWhat is the secret access code for the {city} office? "
                      "Answer with the number only.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
            j = call({"model": "bench", "prompt": prompt, "max_tokens": 12, "temperature": 0})
            ans = j["choices"][0]["text"].strip()
            ok = code in ans
            res["needle"].append({"chars": target_chars, "depth": depth, "ok": ok, "ans": ans[:20],
                                  "prompt_tokens": j["usage"]["prompt_tokens"]})
            print(f"needle tokens={j['usage']['prompt_tokens']:6d} depth={depth:.2f} ok={ok} ans={ans[:20]!r}", flush=True)

acc = sum(r["ok"] for r in res["needle"]) / len(res["needle"])
print(f"needle accuracy {acc:.3f} over {len(res['needle'])}")
json.dump(res, open(sys.argv[1], "w"), indent=1)
