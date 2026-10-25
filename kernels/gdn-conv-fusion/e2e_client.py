"""Greedy completions used to compare VLLM_GDN_FUSED_CONV=0/1 (run inside the server
container). Prompts cover a short request, a ~6k-token prefill, and a concurrent batch
of varied lengths (varlen prefill + decode mixes)."""

import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:8000/v1/completions"
src_dir = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/"
code = open(src_dir + "qwen3_next.py").read()


def req(prompt, n=64):
    body = json.dumps({"model": "bench", "prompt": prompt, "max_tokens": n, "temperature": 0,
                       "seed": 0}).encode()
    r = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    out = json.loads(urllib.request.urlopen(r, timeout=600).read())
    return out["choices"][0]["text"]


prompts = [
    "The capital of France is",
    "Summarize what the following code does.\n\n" + code[:24000] + "\n\nSummary:",
]
batch = [code[i * 3000:(i * 3000) + 800 * (i + 1)] + "\n# Explain the code above:" for i in range(6)]
res = {}
for rep in (1, 2):  # twice: shows whether the server itself is run-to-run deterministic
    res[f"single{rep}"] = [req(p) for p in prompts]
    with ThreadPoolExecutor(6) as ex:
        res[f"batch{rep}"] = list(ex.map(req, batch))
print(json.dumps(res, indent=1))
