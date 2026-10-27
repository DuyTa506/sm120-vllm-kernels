#!/usr/bin/env python3
"""Run repository serving workloads against one existing container/server.

Usage (run under a Docker-enabled account):
  python3 bench/serving_matrix.py --preflight
  python3 bench/serving_matrix.py --container my-server --suite prefill --dry-run
  python3 bench/serving_matrix.py --container my-server --output-dir results/my-run --suite all

The output directory must be mounted as /bench-results in the container.
Each suite creates its own serving_matrix_<suite> subdirectory and refuses to
overwrite an existing suite directory. --base-url addresses host metrics;
--container-base-url addresses the benchmark server inside the container.
Every phase uses a fresh run/scenario/phase cache salt. Mixed output targets
are reconstructed offline with the installed sampler before requests begin.
This script never starts, stops, or reconfigures a server.
"""

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import urllib.request
import uuid


def preflight(runtime_args=None):
    assert callable(globals().get("scenarios")), "scenario matrix not implemented"
    cases = scenarios("all", 0)
    assert len(cases) == 17
    expected = [
        ("long-doc", 32768, 128, 2, 8), ("prefill-heavy", 8192, 64, 8, 48),
        ("rag", 4096, 256, 16, 96), ("shared-prefix", 1024, 256, 16, 96),
        ("mixed-len", 4096, 512, 32, 128), ("chat", 1024, 512, 32, 128),
        ("decode-heavy", 128, 1024, 64, 192),
    ]
    assert [(s.name, s.input_len, s.output_len, s.concurrency, s.prompts)
            for s in cases[:7]] == expected
    assert [(s.input_len, s.output_len, s.concurrency, s.prompts) for s in cases[7:13]] == [
        (length, 1, 1, 8) for length in (2048, 4096, 5000, 8192, 16384, 32768)]
    assert [(s.input_len, s.output_len, s.concurrency, s.prompts) for s in cases[13:]] == [
        (128, 512, c, 8 if c == 1 else c * 6) for c in (1, 4, 16, 64)]
    args = parser().parse_args(["--container", "fixture-server", "--dry-run"])
    seeds, salts = [], []
    for case in cases:
        for warmup in (False, True):
            command = bench_command(args, case, warmup, "/tmp/fixture", "fixture-run")
            value = lambda flag: command[command.index(flag) + 1]
            assert command[:3] == ["docker", "exec", "fixture-server"]
            assert command[3:8] == ["timeout", "--signal=TERM", "--kill-after=15s", "3600s", "vllm"]
            assert command[8:10] == ["bench", "serve"]
            assert value("--num-warmups") == value("--ready-check-timeout-sec") == "0"
            assert int(value("--num-prompts")) == (case.warmups if warmup else case.prompts)
            assert int(value("--random-input-len")) == case.input_len
            assert int(value("--random-output-len")) == case.output_len
            assert int(value("--max-concurrency")) == case.concurrency
            assert "--save-detailed" in command and "--save-result" in command
            seeds.append(int(value("--seed")))
            salt = json.loads(value("--extra-body"))["cache_salt"]
            assert salt == cache_salt("fixture-run", case, warmup)
            assert salt != cache_salt("another-run", case, warmup)
            salts.append(salt)
    assert len(seeds) == len(set(seeds)), "warmup/measured seeds overlap"
    assert len(salts) == len(set(salts)), "cache salts overlap across phases"
    assert cases[3].extra == ("--random-prefix-len", "4096")
    assert cases[4].extra == ("--random-range-ratio", "0.9")
    assert all(s.seed == base.seed + 99 for s, base in zip(scenarios("all", 99), cases))
    for suite, count in (("throughput", 7), ("prefill", 6), ("decode", 4)):
        assert len(scenarios(suite, 0)) == count
    case = cases[0]
    fixture = {"num_prompts": case.prompts, "max_concurrency": case.concurrency,
               "completed": case.prompts, "failed": 0,
               **{field: 1.0 for field in METRICS.values()},
               **{field: [1] * case.prompts for field in DETAILS}}
    fixture["errors"] = [""] * case.prompts
    fixture["output_lens"] = [case.output_len] * case.prompts
    row = result_row(fixture, case, False)
    assert row["scenario"] == case.name and row["completed"] == case.prompts
    mutations = [
        {"completed": case.prompts - 1}, {"failed": 1}, {"num_prompts": 1},
        {"errors": ["request failed"] + [""] * (case.prompts - 1)},
        {"output_lens": [1]}, {"output_lens": [0] * case.prompts},
        {"output_lens": [1] * case.prompts},
        {"output_lens": [case.output_len - 1] + [case.output_len] * (case.prompts - 1)},
        {"output_lens": [case.output_len + 1] * case.prompts},
        {"request_throughput": float("nan")}, {"duration": -1},
        {"max_concurrency": 99},
    ]
    rejected = 0
    for changes in mutations:
        broken = copy.deepcopy(fixture)
        broken.update(changes)
        try:
            result_row(broken, case, False)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError(f"accepted invalid result: {changes}")
    broken = copy.deepcopy(fixture)
    del broken["errors"]
    try:
        result_row(broken, case, False)
    except ValueError:
        rejected += 1
    else:
        raise AssertionError("accepted result without detailed errors")
    mixed = cases[4]
    targets = [51 + (i * 17) % 923 for i in range(mixed.prompts)]
    mixed_fixture = {"num_prompts": mixed.prompts, "max_concurrency": mixed.concurrency,
                     "completed": mixed.prompts, "failed": 0,
                     **{field: 1.0 for field in METRICS.values()},
                     **{field: [1] * mixed.prompts for field in DETAILS},
                     "errors": [""] * mixed.prompts, "output_lens": targets.copy()}
    result_row(mixed_fixture, mixed, False, targets)
    # All these values are inside 51..973; a range-only check would accept
    # them. Exact targets must reject truncation and request reordering.
    for lengths, expected_lengths in (
        ([51] * mixed.prompts, targets),
        (targets[1:] + targets[:1], targets),
        ([targets[0] + 1] + targets[1:], targets),
        (targets, None),
    ):
        broken = copy.deepcopy(mixed_fixture)
        broken["output_lens"] = lengths
        try:
            result_row(broken, mixed, False, expected_lengths)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("accepted malformed/missing mixed-length targets")
    sampling_checked = False
    if runtime_args is not None and runtime_args.container:
        targets_record = reconstruct_expected_lengths(runtime_args, cases)
        for warmup in (True, False):
            key = phase_name(mixed, warmup)
            target = targets_record["cases"][key]
            n = mixed.warmups if warmup else mixed.prompts
            actual_fixture = {"num_prompts": n, "max_concurrency": mixed.concurrency,
                              "completed": n, "failed": 0,
                              **{field: 1.0 for field in METRICS.values()},
                              **{field: [1] * n for field in DETAILS},
                              "errors": [""] * n, "output_lens": target["output_lens"]}
            result_row(actual_fixture, mixed, warmup, target["output_lens"])
        assert targets_record["sampler"]["cuda_initialized"] is False
        if runtime_args.output_dir:
            target_dir = Path(runtime_args.output_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / "serving_matrix_expected_lengths_preflight.json").write_text(
                json.dumps(targets_record, indent=2) + "\n")
        sampling_checked = True
    print(json.dumps({"status": "passed", "scenarios": len(cases),
                      "unique_seeds": len(seeds), "unique_cache_salts": len(salts),
                      "invalid_results_rejected": rejected,
                      "installed_sampling_checked": sampling_checked,
                      "subprocesses": int(sampling_checked), "http_requests": 0, "gpu_work": False}))


@dataclass(frozen=True)
class Scenario:
    suite: str
    name: str
    input_len: int
    output_len: int
    concurrency: int
    prompts: int
    warmups: int
    seed: int
    extra: tuple[str, ...] = ()


def scenarios(suite, seed_offset):
    # Measured shapes/counts mirror throughput_sweep.sh, bench_prefill.sh,
    # and bench_decode.sh. Seeds occupy separate namespaces across suites.
    rows = [
        ("long-doc", 32768, 128, 2, 8, ()),
        ("prefill-heavy", 8192, 64, 8, 48, ()),
        ("rag", 4096, 256, 16, 96, ()),
        ("shared-prefix", 1024, 256, 16, 96, ("--random-prefix-len", "4096")),
        ("mixed-len", 4096, 512, 32, 128, ("--random-range-ratio", "0.9")),
        ("chat", 1024, 512, 32, 128, ()),
        ("decode-heavy", 128, 1024, 64, 192, ()),
    ]
    all_cases = [Scenario("throughput", name, n_in, n_out, c, n, c,
                          seed_offset + 100 + i, extra)
                 for i, (name, n_in, n_out, c, n, extra) in enumerate(rows)]
    all_cases += [Scenario("prefill", f"prefill-{length}", length, 1, 1, 8, 2,
                           seed_offset + 20_000 + length)
                  for length in (2048, 4096, 5000, 8192, 16384, 32768)]
    # Decode warmups cover the target concurrency with disjoint prompts; the
    # original two CLI warmups reuse one prompt and cannot preheat that shape.
    all_cases += [Scenario("decode", f"decode-c{c}", 128, 512, c,
                           8 if c == 1 else 6 * c, max(2, c), seed_offset + 40_000 + c)
                  for c in (1, 4, 16, 64)]
    return [s for s in all_cases if suite == "all" or s.suite == suite]


def phase_name(case, warmup):
    return f"{case.name}.{'warmup' if warmup else 'measured'}"


def cache_salt(run_nonce, case, warmup):
    # One salt per phase retains the shared-prefix workload's intended reuse
    # while separating incidental collisions across scenarios and prior runs.
    return f"serving-matrix:{run_nonce}:{case.suite}:{phase_name(case, warmup)}"


def bench_command(args, case, warmup, remote_dir, run_nonce):
    phase = "warmup" if warmup else "measured"
    return [
        "docker", "exec", args.container,
        "timeout", "--signal=TERM", "--kill-after=15s", f"{args.timeout}s",
        "vllm", "bench", "serve",
        "--backend", "vllm", "--base-url", args.container_base_url,
        "--model", args.model, "--served-model-name", args.served_model_name,
        "--dataset-name", "random", "--ignore-eos", "--disable-tqdm",
        "--num-warmups", "0", "--ready-check-timeout-sec", "0",
        "--extra-body", json.dumps({"cache_salt": cache_salt(run_nonce, case, warmup)}),
        "--save-result", "--save-detailed", "--result-dir", remote_dir,
        "--result-filename", f"{case.name}.{phase}.json",
        "--random-input-len", str(case.input_len),
        "--random-output-len", str(case.output_len),
        "--max-concurrency", str(case.concurrency),
        "--num-prompts", str(case.warmups if warmup else case.prompts),
        "--seed", str(case.seed + (1_000_000 if warmup else 0)),
        *case.extra,
    ]


# Run inside the same container as vllm bench, with CUDA hidden and Hugging
# Face offline. The real tokenizer is necessary because the preceding input
# RNG draw uses input_len - num_special_tokens_to_add(). RandomDataset seeds
# default_rng directly and calls get_sampling_params before offsets/prefix
# generation or prompt tokenization. The helper retains that exact draw order.
_SAMPLING_HELPER = r'''
import hashlib
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import torch
from vllm.benchmarks.datasets.utils import get_sampling_params
from vllm.benchmarks.datasets.datasets import RandomDataset
from vllm.tokenizers import get_tokenizer
model, specs_json = sys.argv[1:]
tokenizer = get_tokenizer(model)
specs = json.loads(specs_json)
records = {}
for spec in specs:
    input_lens, output_lens, offsets = get_sampling_params(
        np.random.default_rng(spec["seed"]), spec["count"], spec["range_ratio"],
        spec["input_len"], spec["output_len"], tokenizer,
    )
    records[spec["name"]] = {**spec, "sampled_input_lens": input_lens.tolist(),
                            "output_lens": output_lens.tolist()}
assert not torch.cuda.is_initialized()
files = [Path(inspect.getfile(get_sampling_params)), Path(inspect.getfile(RandomDataset))]
record = {"cases": records, "sampler": {
    "model": model, "numpy_version": np.__version__,
    "tokenizer_class": type(tokenizer).__name__,
    "num_special_tokens": int(tokenizer.num_special_tokens_to_add()),
    "vocab_size": int(tokenizer.vocab_size), "cuda_initialized": False,
    "draw_order": ["input_lens", "output_lens", "offsets", "prefix_after_sampling"],
    "sources_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
}}
print("SAMPLING_JSON=" + json.dumps(record))
'''


def reconstruct_expected_lengths(args, cases, log_path=None):
    records, mixed_specs = {}, []
    for case in cases:
        for warmup in (True, False):
            name = phase_name(case, warmup)
            count = case.warmups if warmup else case.prompts
            spec = {"name": name, "count": count,
                    "seed": case.seed + (1_000_000 if warmup else 0),
                    "input_len": case.input_len, "output_len": case.output_len}
            if "--random-range-ratio" in case.extra:
                spec["range_ratio"] = float(case.extra[case.extra.index("--random-range-ratio") + 1])
                mixed_specs.append(spec)
            else:
                records[name] = {**spec, "range_ratio": 0.0,
                                 "output_lens": [case.output_len] * count}
    sampler = None
    if mixed_specs:
        command = ["docker", "exec", "-e", "CUDA_VISIBLE_DEVICES=",
                   "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
                   args.container, "timeout", "--signal=TERM", "--kill-after=5s", "120s",
                   "python3", "-c", _SAMPLING_HELPER, args.model, json.dumps(mixed_specs)]
        completed = subprocess.run(command, capture_output=True, text=True,
                                   timeout=135, check=False)
        if log_path is not None:
            log_path.write_text(completed.stderr + "\n" + completed.stdout)
        if completed.returncode:
            raise RuntimeError(f"offline sampling reconstruction failed ({completed.returncode}): {completed.stderr[-2000:]}")
        lines = [line.removeprefix("SAMPLING_JSON=") for line in completed.stdout.splitlines()
                 if line.startswith("SAMPLING_JSON=")]
        if len(lines) != 1:
            raise ValueError("offline sampler did not emit one complete target record")
        sampled = json.loads(lines[0])
        sampler = sampled["sampler"]
        if sampler["cuda_initialized"] is not False:
            raise ValueError("offline sampler initialized CUDA")
        for spec in mixed_specs:
            record = sampled["cases"][spec["name"]]
            if any(record.get(key) != value for key, value in spec.items()):
                raise ValueError("offline sampler changed scenario/seed/count parameters")
            values = record["output_lens"]
            if len(values) != spec["count"] or any(type(n) is not int or n <= 0 for n in values):
                raise ValueError("offline sampler returned incomplete output-length targets")
            records[spec["name"]] = record
    return {"cases": records, "sampler": sampler}


METRICS = {
    "duration_s": "duration", "req_s": "request_throughput",
    "total_tok_s": "total_token_throughput", "out_tok_s": "output_throughput",
    "mean_ttft_ms": "mean_ttft_ms", "med_ttft_ms": "median_ttft_ms",
    "p99_ttft_ms": "p99_ttft_ms", "med_tpot_ms": "median_tpot_ms",
    "med_itl_ms": "median_itl_ms",
}
DETAILS = ("input_lens", "output_lens", "ttfts", "itls", "latencies",
           "start_times", "queue_times", "generated_texts", "errors")


def result_row(result, case, warmup, expected_output_lens=None):
    expected = case.warmups if warmup else case.prompts
    for field, value in (("num_prompts", expected), ("completed", expected),
                         ("failed", 0), ("max_concurrency", case.concurrency)):
        if type(result.get(field)) is not int or result[field] != value:
            raise ValueError(f"{case.name}: {field}={result.get(field)!r}, expected {value}")
    for field in DETAILS:
        values = result.get(field)
        if not isinstance(values, list) or len(values) != expected:
            raise ValueError(f"{case.name}: missing/incomplete detailed {field}")
    if any(not isinstance(error, str) or error for error in result["errors"]):
        raise ValueError(f"{case.name}: per-request errors present")
    for field in ("input_lens", "output_lens"):
        if any(type(n) is not int or n <= 0 for n in result[field]):
            raise ValueError(f"{case.name}: invalid/incomplete {field}")
    if "--random-range-ratio" in case.extra:
        wanted = expected_output_lens
        if not isinstance(wanted, list) or len(wanted) != expected or any(type(n) is not int or n <= 0 for n in wanted):
            raise ValueError(f"{case.name}: exact sampled output-length targets are required")
    else:
        wanted = [case.output_len] * expected
    mismatches = [(i, got, want) for i, (got, want) in enumerate(zip(result["output_lens"], wanted)) if got != want]
    if mismatches:
        raise ValueError(f"{case.name}: truncated/incorrect output lengths (request, actual, expected): {mismatches[:5]}")
    row = {"suite": case.suite, "scenario": case.name,
           "in": case.input_len, "out": case.output_len, "conc": case.concurrency,
           "num_prompts": expected, "seed": case.seed + (1_000_000 if warmup else 0),
           "completed": result["completed"], "failed": result["failed"]}
    for column, field in METRICS.items():
        value = result.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{case.name}: missing/nonfinite/negative {field}")
        if field in ("duration", "request_throughput", "output_throughput", "total_token_throughput") and value <= 0:
            raise ValueError(f"{case.name}: zero {field} with successful requests")
        row[column] = value
    return row


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_summary(directory, status, rows, error=None):
    summary = {"status": status, "updated_at": utcnow(), "results": rows, "error": error}
    temporary = directory / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    temporary.replace(directory / "summary.json")
    columns = ["suite", "scenario", "in", "out", "conc", "num_prompts", "seed",
               "completed", "failed", *METRICS]
    temporary = directory / "summary.tsv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(directory / "summary.tsv")


def run(args):
    cases = scenarios(args.suite, args.seed_offset)
    run_nonce = uuid.uuid4().hex
    remote_dir = args.container_output_dir.rstrip("/") + f"/serving_matrix_{args.suite}"
    commands = [{"scenario": case.name, "warmup": warmup,
                 "cache_salt": cache_salt(run_nonce, case, warmup),
                 "command": bench_command(args, case, warmup, remote_dir, run_nonce)}
                for case in cases for warmup in (True, False)]
    manifest = {"created_at": utcnow(), "run_nonce": run_nonce, "arguments": vars(args),
                "expected_lengths_file": "expected_output_lengths.json",
                "scenarios": [asdict(case) for case in cases], "commands": commands,
                "container_result_dir": remote_dir}
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    directory = Path(args.output_dir) / f"serving_matrix_{args.suite}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    monitor = None
    monitor_log = None
    status, error = "running", None
    write_summary(directory, status, rows)

    def event(**fields):
        with (directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": utcnow(), **fields}) + "\n")

    def command(argv, filename, timeout=60):
        event(kind="command_start", argv=argv, log=filename)
        with (directory / filename).open("w") as stream:
            completed = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=timeout, check=False)
        event(kind="command_end", returncode=completed.returncode, log=filename)
        if completed.returncode:
            raise RuntimeError(f"command failed ({completed.returncode}); see {filename}")

    def metrics(filename):
        url = args.base_url.rstrip("/") + "/metrics"
        event(kind="metrics", url=url, file=filename)
        with urllib.request.urlopen(url, timeout=20) as response:
            (directory / filename).write_bytes(response.read())

    gpu_command = ["nvidia-smi", "-i", args.gpu,
                   "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu,clocks.sm,power.draw,temperature.gpu",
                   "--format=csv"]
    try:
        command(["docker", "exec", args.container, "test", "-d", remote_dir], "mount_check.log")
        targets = reconstruct_expected_lengths(args, cases, directory / "sampling_helper.log")
        (directory / "expected_output_lengths.json").write_text(json.dumps(targets, indent=2) + "\n")
        command(gpu_command, "gpu_before.csv")
        monitor_log = (directory / "gpu_during.csv").open("w")
        monitor = subprocess.Popen([*gpu_command, "--loop=1"], stdout=monitor_log, stderr=subprocess.STDOUT)
        event(kind="telemetry_start", pid=monitor.pid)
        for case in cases:
            for warmup in (True, False):
                phase = "warmup" if warmup else "measured"
                name = f"{case.name}.{phase}"
                if monitor.poll() is not None:
                    raise RuntimeError("GPU telemetry stopped; see gpu_during.csv")
                print(f"{utcnow()} {case.suite}/{name}", flush=True)
                metrics(f"{name}.metrics_before.txt")
                benchmark_failed = False
                try:
                    # Bound the process inside the container as well as the
                    # Docker client, so a timed-out request generator stops.
                    command(bench_command(args, case, warmup, remote_dir, run_nonce), f"{name}.log", args.timeout + 30)
                except BaseException:
                    benchmark_failed = True
                    raise
                finally:
                    # Keep the benchmark failure as the primary error while
                    # retaining evidence if the server also stops responding.
                    try:
                        metrics(f"{name}.metrics_after.txt")
                    except Exception as exc:
                        event(kind="metrics_failure", scenario=name, error=str(exc))
                        if not benchmark_failed:
                            raise
                path = directory / f"{name}.json"
                if not path.is_file():
                    raise RuntimeError(f"missing {path}; verify the output directory mount")
                row = result_row(json.loads(path.read_text()), case, warmup, targets["cases"][name]["output_lens"])
                event(kind="validated", scenario=name, completed=row["completed"], failed=row["failed"])
                if not warmup:
                    rows.append(row)
                    write_summary(directory, "running", rows)
        if monitor.poll() is not None:
            raise RuntimeError("GPU telemetry stopped; see gpu_during.csv")
        command(gpu_command, "gpu_after.csv")
        status = "passed"
    except BaseException as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        event(kind="failure", error=error)
        raise
    finally:
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait(timeout=5)
            event(kind="telemetry_stop", returncode=monitor.returncode)
        if monitor_log is not None:
            monitor_log.close()
        write_summary(directory, status, rows, error)
    print(f"Validated {len(rows)} scenarios; results: {directory}", flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="CPU checks; optionally --container verifies installed sampling offline and --output-dir saves targets")
    mode.add_argument("--dry-run", action="store_true", help="print scenario/command manifest only")
    result.add_argument("--container")
    result.add_argument("--output-dir", help="host directory mounted at --container-output-dir")
    result.add_argument("--container-output-dir", default="/bench-results")
    result.add_argument("--base-url", default="http://127.0.0.1:8132", help="host metrics URL")
    result.add_argument("--container-base-url", default="http://127.0.0.1:8000", help="server URL inside container")
    result.add_argument("--suite", choices=("throughput", "prefill", "decode", "all"), default="all")
    result.add_argument("--model", default="unsloth/Qwen3.6-35B-A3B-NVFP4-Fast")
    result.add_argument("--served-model-name", default="bench")
    result.add_argument("--seed-offset", type=int, default=0)
    result.add_argument("--gpu", default="0", help="nvidia-smi GPU index or UUID for telemetry")
    result.add_argument("--timeout", type=int, default=3600, help="seconds allowed per benchmark invocation")
    return result


def main():
    cli = parser()
    args = cli.parse_args()
    if args.preflight:
        preflight(args)
        return
    if not args.container or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.container):
        cli.error("--container must name an existing Docker container")
    if not args.dry_run and not args.output_dir:
        cli.error("--output-dir is required for execution")
    if args.timeout <= 0 or not 0 <= args.seed_offset <= 1_000_000_000:
        cli.error("timeout must be positive and seed-offset must be in 0..1000000000")
    if not args.container_output_dir.startswith("/"):
        cli.error("--container-output-dir must be absolute")
    run(args)


if __name__ == "__main__":
    main()
