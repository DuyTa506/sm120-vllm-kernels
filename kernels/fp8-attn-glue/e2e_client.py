"""Greedy completion on a fixed ~8k-token prompt; prints JSON with tokens, logprobs and
prompt logprobs so two server configs can be compared (see e2e_compare.py)."""

import argparse
import json
import random
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--out", default="/dev/stdout")
ap.add_argument("--varlen", action="store_true")
args = ap.parse_args()

rng = random.Random(1234)
words = ("the model attention kernel layer prefill decode token cache page block value key "
         "query softmax scale tensor memory bandwidth latency throughput expert router gate "
         "linear state chunk sequence length batch graph stream warp thread register").split()
paras = []
for i in range(420):
    paras.append(f"Section {i}: " + " ".join(rng.choice(words) for _ in range(14)) + ".")
prompt = ("Read the following notes and then answer the question at the end.\n\n" + "\n".join(paras)
          + "\n\nQuestion: which section number appears first in the notes, and summarize the "
          "notes in two sentences.\nAnswer:")

def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


if args.varlen:
    # concurrent prompts of different lengths -> varlen prefill batches (+ decodes) in the
    # server; used with VLLM_SM120_FP8_PREFILL_NOGLUE_VERIFY=1 to check every layer call.
    import threading

    outs = []

    def one(n, seed):
        r2 = random.Random(seed)
        p = " ".join(r2.choice(words) for _ in range(n))
        outs.append(post({"model": "bench", "prompt": p, "max_tokens": 16, "temperature": 0.0}))

    th = [threading.Thread(target=one, args=(n, i)) for i, n in
          enumerate([300, 2500, 700, 5200, 1200, 9000, 450, 3100])]
    for t in th:
        t.start()
    for t in th:
        t.join()
    print(json.dumps({"varlen_requests": len(outs)}))
    raise SystemExit(0)

res = {}
for name, body in (
    ("gen", {"model": "bench", "prompt": prompt, "max_tokens": 48, "temperature": 0.0,
             "logprobs": 5}),
    ("prompt_lp", {"model": "bench", "prompt": prompt, "max_tokens": 1, "temperature": 0.0,
                   "prompt_logprobs": 0}),
    ("prompt_lp_repeat", {"model": "bench", "prompt": prompt, "max_tokens": 1,
                          "temperature": 0.0, "prompt_logprobs": 0}),
):
    r = post(body)
    c = r["choices"][0]
    if name == "gen":
        res["text"] = c["text"]
        res["prompt_tokens"] = r["usage"]["prompt_tokens"]
        res["token_logprobs"] = c["logprobs"]["token_logprobs"]
        res["tokens"] = c["logprobs"]["tokens"]
    else:
        plp = c.get("prompt_logprobs") or []
        vals = []
        for d in plp:
            if not d:
                vals.append(None)
                continue
            v = next(iter(d.values()))
            vals.append(v["logprob"] if isinstance(v, dict) else v)
        res["prompt_logprobs" if name == "prompt_lp" else "prompt_logprobs_repeat"] = vals
with open(args.out, "w") as f:
    json.dump(res, f)
