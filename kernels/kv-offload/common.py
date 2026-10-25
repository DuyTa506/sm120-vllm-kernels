"""Shared client helpers for the kv-offload experiments (stdlib only)."""
import json, os, time, urllib.request

URL = os.environ.get("BENCH_URL", "http://127.0.0.1:8121")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
LONG_TRAJ = os.path.join(REPO, "results/agentic/trajectories_long_production.json")


def post(path, body=None, stream=False, timeout=1800):
    data = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(URL + path, data, {"Content-Type": "application/json"}, method="POST")
    r = urllib.request.urlopen(req, timeout=timeout)
    return r if stream else json.loads(r.read() or b"{}")


def metrics():
    """All counters/gauges/histogram sums of /metrics, summed over label sets."""
    txt = urllib.request.urlopen(URL + "/metrics", timeout=30).read().decode()
    out = {}
    for line in txt.splitlines():
        if not line or line[0] == "#":
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[-1])
        except ValueError:
            pass
    return out


def mdelta(m0, m1, names):
    return {n: m1.get(n, 0.0) - m0.get(n, 0.0) for n in names}


KEY_METRICS = [
    "vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total",
    "vllm:external_prefix_cache_queries_total", "vllm:external_prefix_cache_hits_total",
    "vllm:num_preemptions_total", "vllm:prompt_tokens_total", "vllm:generation_tokens_total",
    "vllm:kv_offload_load_bytes_total", "vllm:kv_offload_load_time_total", "vllm:kv_offload_load_size_total",
    "vllm:kv_offload_store_bytes_total", "vllm:kv_offload_store_time_total", "vllm:kv_offload_store_size_total",
    "vllm:kv_offload_allocation_failure_total",
    "vllm:simple_kv_offload_load_blocks_total", "vllm:simple_kv_offload_save_outcomes_total",
]


def complete(prompt, salt, max_tokens, logprobs=None, min_tokens=None, ignore_eos=False, on_first=None):
    """Streaming /v1/completions with token-id prompt. Returns dict with ttft, e2e, text, usage, lps."""
    body = {"model": "bench", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "stream": True,
            "cache_salt": salt, "stream_options": {"include_usage": True}}
    if min_tokens is not None:
        body["min_tokens"] = min_tokens
    if ignore_eos:
        body["ignore_eos"] = True
    if logprobs is not None:
        body["logprobs"] = logprobs
    t0 = time.time(); ttft = None; text = []; usage = None; lps = []
    r = post("/v1/completions", body, stream=True)
    for line in r:
        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
            continue
        j = json.loads(line[6:])
        if j.get("usage"):
            usage = j["usage"]
        for ch in j.get("choices", []):
            if ttft is None:
                ttft = time.time() - t0
                if on_first:
                    on_first(ttft)
            text.append(ch.get("text", ""))
            lp = ch.get("logprobs")
            if lp and lp.get("top_logprobs"):
                lps.extend(lp["top_logprobs"])
    cached = None
    if usage and usage.get("prompt_tokens_details"):
        cached = usage["prompt_tokens_details"].get("cached_tokens")
    return {"ttft": ttft, "e2e": time.time() - t0, "text": "".join(text), "usage": usage, "cached": cached,
            "top_logprobs": lps}


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    return xs[min(int(len(xs) * p), len(xs) - 1)]
