"""Explicit mixed-arrival streaming workload using only Python stdlib.

Token IDs are deterministic synthetic inputs, not a quality benchmark.
"""
import argparse
import ast
import hashlib
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import threading
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8132")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decode-concurrency", type=int, default=16)
    parser.add_argument("--seed", type=int, default=73019)
    parser.add_argument("--synthetic-stress", action="store_true")
    args = parser.parse_args()
    assert 1 <= args.decode_concurrency <= 32
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    def literal(path, name):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                return ast.literal_eval(node.value)
        raise ValueError(f"missing literal {name}")
    source_paths = [root / "eval/quality_probe.py", root / "bench/agentic_bench.py", root / "kernels/moe-nvfp4/data/prompt.txt"]
    qa = literal(source_paths[0], "prompts")
    code = literal(source_paths[1], "TASKS")
    document = source_paths[2].read_text()
    def tokenize(text):
        req = urllib.request.Request(args.url + "/tokenize", data=json.dumps(dict(model="bench", prompt=text)).encode(), headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response)["tokens"]
    prepared = {}
    if not args.synthetic_stress:
        for i in range(args.decode_concurrency):
            task = code[(i // 2) % len(code)] if i % 2 == 0 else qa[(i // 2) % len(qa)]
            text = "<|im_start|>user\n" + task + "\nAnswer directly with a self-contained explanation or implementation; tools are unavailable.<|im_end|>\n<|im_start|>assistant\n"
            prepared[i] = tokenize(text)
        suffix = tokenize("\nSummarize this document and explain its major provisions.\n<|im_end|>\n<|im_start|>assistant\n")
        document_ids = tokenize("<|im_start|>user\n" + document)
        assert len(document_ids) >= 32768 - len(suffix)
        for i in range(2):
            prepared[100 + i] = document_ids[:32768-len(suffix)] + suffix
    (args.output / "input_manifest.json").write_text(json.dumps(dict(
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        prompt_sha256={str(k):hashlib.sha256(json.dumps(v).encode()).hexdigest() for k,v in prepared.items()},
        regime="synthetic_stress" if args.synthetic_stress else "repository_code_qa_document"), indent=2))
    starts = [threading.Event() for _ in range(args.decode_concurrency)]

    def metrics(name):
        with urllib.request.urlopen(args.url + "/metrics", timeout=30) as response:
            (args.output / name).write_bytes(response.read())

    def request(index, kind, length, output, event=None):
        rng = random.Random(args.seed + index)
        prompt_ids = ([rng.randrange(1000, 20000) for _ in range(length)] if args.synthetic_stress else prepared[index])
        payload = dict(model="bench", prompt=prompt_ids, cache_salt=f"mtp-{args.seed}-{index}",
                       max_tokens=output, temperature=0, stream=True, ignore_eos=args.synthetic_stress,
                       stream_options=dict(include_usage=True))
        req = urllib.request.Request(args.url + "/v1/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        begin = time.monotonic()
        arrivals, usage = [], None
        with urllib.request.urlopen(req, timeout=1200) as response:
            for raw in response:
                if not raw.startswith(b"data: ") or raw.strip() == b"data: [DONE]":
                    continue
                data = json.loads(raw[6:])
                if "error" in data:
                    raise RuntimeError(data["error"])
                if data.get("usage"):
                    usage = data["usage"]
                if any(choice.get("text") for choice in data.get("choices", [])):
                    arrivals.append(time.monotonic())
                    if event:
                        event.set()
        if not arrivals or usage is None or not 0 < usage["completion_tokens"] <= output:
            raise RuntimeError(f"incomplete request {index}: {usage}")
        return dict(index=index, kind=kind, input_tokens=len(prompt_ids), requested_output_tokens=output,
                    begin=begin, end=time.monotonic(), arrivals=arrivals, usage=usage,
                    ttft_ms=1000 * (arrivals[0] - begin),
                    stream_chunk_gaps_ms=[1000 * (b-a) for a,b in zip(arrivals, arrivals[1:])])

    metrics("metrics_before.txt")
    with ThreadPoolExecutor(max_workers=args.decode_concurrency + 2) as pool:
        decode = [pool.submit(request, i, "decode", 128, 1024, event)
                  for i, event in enumerate(starts)]
        for event in starts:
            if not event.wait(180):
                raise RuntimeError("decode did not reach first streamed text before prefill arrival")
        long_doc = [pool.submit(request, 100 + i, "long-doc", 32768, 128) for i in range(2)]
        rows = [future.result() for future in decode + long_doc]
    metrics("metrics_after.txt")
    overlap = sum(any(d["begin"] < p["begin"] < d["end"] for d in rows if d["kind"] == "decode")
                  for p in rows if p["kind"] == "long-doc")
    result = dict(seed=args.seed, decode_concurrency=args.decode_concurrency, rows=rows,
                  prefill_arrivals_overlapping_decode=overlap,
                  caveat="SSE chunks may contain multiple accepted tokens; chunk gaps are not per-token ITL")
    (args.output / "requests.json").write_text(json.dumps(result, indent=2))
    if overlap != 2:
        raise SystemExit("invalid mixed-load trial: long prefills did not overlap decode")


if __name__ == "__main__":
    main()
