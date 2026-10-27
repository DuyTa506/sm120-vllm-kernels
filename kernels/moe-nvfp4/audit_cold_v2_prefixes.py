#!/usr/bin/env python3
"""CPU-only prefix-overlap audit for the completed cold-v2 A3/B2 schedule.

Uses the installed pure sampling/helper functions via AST extraction and the
actual snapshot tokenizer. Does not import vLLM/torch or contact a server.

Run inside qwen-prefill-dev with this repository mounted at /work:
  python3 /work/kernels/moe-nvfp4/audit_cold_v2_prefixes.py

JSON is printed to stdout. --output optionally writes the same report.
"""

import argparse
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

SNAPSHOT = "1c3f884bc99aac2524f6d49bcbac8c88401afd66"
MODEL = "unsloth/Qwen3.6-35B-A3B-NVFP4-Fast"


def schedule():
    phases = []
    for length in (8192, 16384):
        for seed in (101, 202):
            phases.extend([
                (f"warmup_{length}_{seed}", length, 1, 2, length + seed + 100000, True),
                (f"prefill_{length}_{seed}", length, 1, 16, length + seed, False),
            ])
    phases.extend([
        ("decode_warmup_c1", 128, 64, 2, 1501, True),
        ("decode_c1", 128, 256, 8, 501, False),
        ("decode_warmup_c4", 128, 64, 8, 1504, True),
        ("decode_c4", 128, 256, 16, 504, False),
        ("profile_warmup", 8192, 1, 2, 1777, True),
        ("profile_8k", 8192, 1, 4, 777, False),
    ])
    return phases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path(
        "/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots"
    ) / SNAPSHOT)
    parser.add_argument("--vllm-root", type=Path, default=Path(
        "/usr/local/lib/python3.12/dist-packages/vllm"
    ))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    from tokenizers import Tokenizer

    tokenizer_file = args.snapshot / "tokenizer.json"
    config = json.loads(tokenizer_file.read_text())
    tok = Tokenizer.from_file(str(tokenizer_file))

    class Adapter:
        vocab_size = tok.get_vocab_size(with_added_tokens=False)
        all_special_ids = [v["id"] for v in config["added_tokens"] if v.get("special", False)]

        def num_special_tokens_to_add(self):
            return tok.num_special_tokens_to_add(False)

        def decode(self, ids):
            return tok.decode(ids, skip_special_tokens=False)

        def encode(self, text, add_special_tokens=False):
            return tok.encode(text, add_special_tokens=add_special_tokens).ids

    class Logger:
        def info(self, *args):
            pass

    adapter = Adapter()
    namespace = {"TokenizerLike": object, "np": np, "math": math,
                 "RangeRatio": float | dict[str, float], "logger": Logger()}
    source_hashes = {}

    def extract(path, names):
        source = path.read_text()
        source_hashes[str(path)] = hashlib.sha256(source.encode()).hexdigest()
        tree = ast.parse(source)
        funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
        if {f.name for f in funcs} != names:
            raise RuntimeError(f"Missing installed helper functions in {path}")
        module = ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[]))
        exec(compile(module, str(path), "exec"), namespace)

    extract(args.vllm_root / "benchmarks/datasets/utils.py",
            {"_resolve_range_ratios", "get_sampling_params"})
    extract(args.vllm_root / "benchmarks/datasets/datasets.py",
            {"gen_prompt_decode_to_target_len"})
    allowed = np.array(list(set(np.arange(adapter.vocab_size)) - set(adapter.all_special_ids)))
    phases = schedule()
    records, length_mismatches = [], []
    starts = defaultdict(list)
    for phase, length, output_length, count, seed, warmup in phases:
        rng = np.random.default_rng(seed)
        inputs, outputs, offsets = namespace["get_sampling_params"](
            rng, count, 0.0, length, output_length, adapter)
        for index in range(count):
            start = (int(offsets[index]) + index) % len(allowed)
            initial = allowed[(int(offsets[index]) + index + np.arange(int(inputs[index]))) % len(allowed)].tolist()
            prompt, _, mismatch = namespace["gen_prompt_decode_to_target_len"](
                adapter, initial, int(inputs[index]), rng=rng)
            actual = adapter.encode(prompt, add_special_tokens=False)
            identity = {"phase": phase, "request_index": index, "seed": seed, "warmup": warmup}
            records.append((identity, actual))
            starts[start].append(identity)
            if mismatch or len(actual) != int(inputs[index]):
                length_mismatches.append({**identity, "expected": int(inputs[index]),
                                          "actual": len(actual), "helper_mismatch": mismatch})
            if int(outputs[index]) != output_length:
                raise AssertionError("Fixed output sampling changed")

    maximum, maximum_pair, overlaps = 0, None, []
    for index, (a, a_tokens) in enumerate(records):
        for b, b_tokens in records[index + 1:]:
            common = 0
            for a_token, b_token in zip(a_tokens, b_tokens):
                if a_token != b_token:
                    break
                common += 1
            if common > maximum:
                maximum, maximum_pair = common, [a, b]
            if common >= 16:
                overlaps.append({"requests": [a, b], "prefix_tokens": common,
                                 "cross_phase": a["phase"] != b["phase"]})
    count = len(records)
    report = {
        "status": "passed" if not overlaps and not length_mismatches else "overlap_or_length_mismatch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "cold-v2", "runs": ["shared_a3", "shared_b2"],
        "model": MODEL, "tokenizer_snapshot": args.snapshot.name,
        "tokenizer_file": str(tokenizer_file),
        "tokenizer_sha256": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
        "vocab_size": adapter.vocab_size,
        "special_added_per_prompt": adapter.num_special_tokens_to_add(),
        "phase_count": len(phases), "request_count_per_run": count,
        "warmup_request_count_per_run": sum(p[3] for p in phases if p[5]),
        "measured_request_count_per_run": sum(p[3] for p in phases if not p[5]),
        "pair_count_per_run": count * (count - 1) // 2,
        "raw_start_collisions": [v for v in starts.values() if len(v) > 1],
        "token_length_mismatches": length_mismatches,
        "actual_prefix_overlaps_16plus": overlaps,
        "maximum_common_prefix_tokens": maximum, "maximum_common_prefix_pair": maximum_pair,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "cuda_initialized": None,
        "cuda_status_note": "CUDA initialization was not queried; CUDA hidden, no torch/vLLM import or GPU work.",
        "torch_imported": "torch" in sys.modules, "gpu_work": False, "http_requests": 0,
        "schedule": [{"phase": name, "input_len": n_in, "output_len": n_out,
                      "num_prompts": n, "seed": seed, "warmup": warmup}
                     for name, n_in, n_out, n, seed, warmup in phases],
    }
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
