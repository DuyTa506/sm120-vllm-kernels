"""Verifier: in-model check of the skinny FP8 patch (run once with VLLM_SM120_SKINNY_FP8=0 and once with =1,
patch files bind-mounted in both runs so only the flag differs).

1) Decode step time that is NOT confounded by MTP acceptance: wall time of the decode phase divided by
   the number of engine decode steps (= spec-decode 'num_drafts' counter delta). Greedy, random-token
   prompts of 128 tokens, 384 output tokens, ignore_eos. c=1 (requests one at a time) and c=4.
   Rounds alternate c=1/c=4 and are repeated; median per-round step time is reported.
2) Optional nsys window (cudaProfilerStart/Stop) around one short c=1 and c=4 generation so the actual
   kernels used inside the CUDA graphs can be listed (run under nsys --capture-range=cudaProfilerApi).
3) Greedy output tokens are dumped so off/on can be compared (numerical drift at the token level).
"""
import json
import os
import statistics
import sys
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # engine in-process: profiler range + metrics
import torch  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

TAG = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else f"/v/insitu_{TAG}.json"
ROUNDS = int(os.environ.get("ROUNDS", "3"))
SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 128, 192, 256, 512]

llm = LLM(
    model="unsloth/Qwen3.6-35B-A3B-NVFP4-Fast",
    language_model_only=True,
    max_model_len=4096,
    max_num_seqs=64,
    gpu_memory_utilization=0.78,
    enable_prefix_caching=False,  # identical prompts across rounds must not hit the cache
    enable_chunked_prefill=True,
    speculative_config={"method": "mtp", "num_speculative_tokens": 2},
    async_scheduling=True,
    cudagraph_capture_sizes=SIZES,
    disable_log_stats=False,
    seed=0,
)

g = torch.Generator().manual_seed(1234)
prompts = [TokensPrompt(prompt_token_ids=torch.randint(1000, 100000, (128,), generator=g).tolist())
           for _ in range(8)]
sp = SamplingParams(temperature=0.0, max_tokens=384, ignore_eos=True)


def counter(name):
    for m in llm.get_metrics():
        if m.name == name:
            return m.value
    return None


def run(batch):
    d0 = counter("vllm:spec_decode_num_drafts")
    a0 = counter("vllm:spec_decode_num_accepted_tokens")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(batch, sp, use_tqdm=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    d = counter("vllm:spec_decode_num_drafts") - d0
    acc = counter("vllm:spec_decode_num_accepted_tokens") - a0
    return dt, d, acc, [list(o.outputs[0].token_ids) for o in outs]


# warmup
run(prompts[:1])
run(prompts[:4])

res = {"c1": [], "c4": [], "tokens": {}}
for r in range(ROUNDS):
    # c=1: 4 requests sequentially; per-step time = wall / steps (drafts counter counts per request-step)
    tot, steps, acc = 0.0, 0, 0
    for i in range(4):
        dt, d, a, toks = run([prompts[i]])
        tot += dt
        steps += d
        acc += a
        res["tokens"][f"c1_{i}"] = toks[0]
    res["c1"].append(dict(ms_per_step=1e3 * tot / steps, steps=steps, accepted=acc))
    # c=4: 4 requests together; drafts counter counts one per request per step -> steps = d / 4
    dt, d, a, toks = run(prompts[4:8])
    res["c4"].append(dict(ms_per_step=1e3 * dt / (d / 4), steps=d / 4, accepted=a))
    for i, t in enumerate(toks):
        res["tokens"][f"c4_{i}"] = t
    print(f"[{TAG}] round {r}: c=1 {res['c1'][-1]}  c=4 {res['c4'][-1]}", flush=True)

for c in ("c1", "c4"):
    v = [x["ms_per_step"] for x in res[c]]
    res[c + "_median_ms_per_step"] = statistics.median(v)
    print(f"[{TAG}] {c}: median ms/step {statistics.median(v):.4f}  all {[round(x, 4) for x in v]}", flush=True)

if os.environ.get("NSYS_WINDOW") == "1":
    sp2 = SamplingParams(temperature=0.0, max_tokens=48, ignore_eos=True)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    llm.generate([prompts[0]], sp2, use_tqdm=False)
    llm.generate(prompts[4:8], sp2, use_tqdm=False)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

json.dump(res, open(OUT, "w"))
print("DONE", flush=True)
