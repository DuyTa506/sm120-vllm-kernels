"""Offline paired greedy-decode comparison for qwen_experiment.sh result roots.

python3 -I bench/compare_greedy_decode.py BASE CAND --output-prefix results/pair
python3 -I bench/compare_greedy_decode.py --preflight

Derives cases from serving_matrix_decode manifests; does not require the full
17-task suite. JSON and Markdown preserve source hashes and MTP counter deltas.
"""
import argparse
import csv
from dataclasses import asdict
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("greedy_serving_matrix", HERE / "serving_matrix.py")
MATRIX = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MATRIX
SPEC.loader.exec_module(MATRIX)
PKG = "/usr/local/lib/python3.12/dist-packages/vllm/"
EXPERT = "model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py"
ORACLE = "model_executor/layers/fused_moe/oracle/unquantized.py"
COUNTERS = ("num_drafts", "num_draft_tokens", "num_accepted_tokens")
VARIANT_ENV = {"VLLM_CACHE_ROOT", "VLLM_B12X_FIX_SCALES", "VLLM_B12X_SHARED_WORKSPACE",
               "VLLM_B12X_SCALAR_INPUT_SCALE"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def option(argv, key):
    require(argv.count(key) == 1 and argv.index(key) + 1 < len(argv), f"missing/duplicate {key}")
    return argv[argv.index(key) + 1]


def mtp_series(text):
    found = {name: {} for name in COUNTERS}
    for line in text.splitlines():
        for name in COUNTERS:
            prefix = f"vllm:spec_decode_{name}_total"
            if line.startswith(prefix + "{") or line.startswith(prefix + " "):
                key, value = line.split()
                number = float(value)
                require(math.isfinite(number) and number >= 0 and number.is_integer(), "invalid MTP counter")
                require(key not in found[name], "duplicate MTP series")
                found[name][key] = number
    require(all(found.values()), "missing MTP counters")
    return found


def mtp_delta(before, after):
    result = {}
    for name in COUNTERS:
        require(before[name].keys() == after[name].keys(), "MTP series changed during phase")
        changes = [after[name][key] - value for key, value in before[name].items()]
        require(all(x >= 0 for x in changes), "MTP counter reset")
        result[name] = int(sum(changes))
    require(result["num_drafts"] > 0 and result["num_draft_tokens"] >= result["num_drafts"],
            "no valid MTP drafts in measured decode")
    require(result["num_accepted_tokens"] <= result["num_draft_tokens"], "invalid accepted token count")
    result["acceptance_rate"] = result["num_accepted_tokens"] / result["num_draft_tokens"]
    result["acceptance_length"] = 1 + result["num_accepted_tokens"] / result["num_drafts"]
    return result


def source_extras(variant):
    return set() if variant == "next" else {
        f"kernels/moe-nvfp4/{variant}/vllm/{EXPERT}",
        f"kernels/moe-nvfp4/patched/vllm/{ORACLE}",
    }


def load(root):
    root = Path(root)
    hashes = {}

    def text(relative):
        data = (root / relative).read_bytes()
        hashes[relative] = sha(data)
        return data.decode()

    def read(relative):
        return json.loads(text(relative))

    require(text("exit_code.txt").strip() == "0", f"failed/incomplete experiment: {root}")
    config = dict(line.split("=", 1) for line in text("config.txt").splitlines())
    require(config["tag"] == root.name and config["greedy_decode"] == "1", "invalid greedy run identity")
    variant = config["variant"]
    require(variant in ("next", "shared", "shared_exact", "scalar"), "unknown overlay variant")
    image = text("image.txt").strip()
    commit, diff_hash = text("commit.txt").strip(), sha(text("uncommitted.diff").encode())
    containers = read("container.json")
    require(isinstance(containers, list) and len(containers) == 1, "invalid container capture")
    container = containers[0]
    require(container["Name"] == "/qwen-exp-" + config["tag"], "container identity mismatch")
    require(container["Image"] == image, "actual container image differs from captured image")
    command = container["Config"]["Cmd"][:]
    if variant != "next":
        require(option(command, "--moe-backend") == "flashinfer_b12x", "candidate backend mismatch")
        pos = command.index("--moe-backend")
        del command[pos:pos + 2]
    else:
        require("--moe-backend" not in command, "baseline unexpectedly overrides the MoE backend")
    speculative = json.loads(option(command, "--speculative-config"))
    require(speculative.get("method") == "mtp" and speculative.get("num_speculative_tokens", 0) > 0,
            "greedy comparison requires MTP enabled")
    env = dict(value.split("=", 1) for value in container["Config"]["Env"])
    if variant != "next":
        require(env.get("VLLM_B12X_FIX_SCALES") == env.get("VLLM_B12X_SHARED_WORKSPACE") == "1",
                "candidate scale/workspace flags missing")
        require(env.get("VLLM_B12X_SCALAR_INPUT_SCALE", "0") == ("1" if variant == "scalar" else "0"),
                "scalar flag does not match variant")
    identity_env = {k: v for k, v in env.items() if k not in VARIANT_ENV}
    gpu_rows = list(csv.DictReader(io.StringIO(text("gpu_before.csv")), skipinitialspace=True))
    require(len(gpu_rows) == 1, "expected one GPU identity")
    gpu = {key: gpu_rows[0][key].strip() for key in
           ("name", "uuid", "driver_version", "memory.total [MiB]", "power.limit [W]")}

    sources = read("source_manifest.json")
    require(sources["schema_version"] == 1 and sources["variant"] == variant, "source manifest identity mismatch")
    for relative, digest in sources["files"].items():
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts, "unsafe source path")
        require(sha(text("source_files/" + relative).encode()) == digest, "source snapshot hash mismatch")
    verification = json.loads(text("source_verify.log"))
    require(verification.get("status") == "passed" and verification.get("action") == "verify"
            and verification.get("variant") == variant
            and verification.get("files") == len(sources["files"]), "source verification failed/missing")
    require(source_extras(variant) <= sources["files"].keys(), "overlay source not captured")
    for mount in container["Mounts"]:
        if mount["Destination"].startswith(PKG) and mount["Destination"].endswith(".py"):
            require(mount["RW"] is False, "runtime Python source was mounted writable")
            matches = [p for p in sources["files"] if mount["Source"].endswith("/" + p)]
            require(len(matches) == 1, "mounted source absent/ambiguous in snapshot")
    for relative in ("scripts/serve.sh", "scripts/cudagraph_sizes.txt", "bench/qwen_experiment.sh",
                     "bench/serving_matrix.py"):
        require(relative in sources["files"], f"missing runtime source {relative}")

    directory = "serving/serving_matrix_decode/"
    manifest, summary = read(directory + "manifest.json"), read(directory + "summary.json")
    targets = read(directory + "expected_output_lengths.json")
    args = manifest["arguments"]
    require(args["suite"] == "decode" and args.get("temperature") == 0, "not explicit greedy decode")
    require(args["container"] == container["Name"].lstrip("/"), "request/container identity mismatch")
    require(summary["status"] == "passed" and not summary.get("error"), "decode matrix failed")
    cases = [MATRIX.Scenario(**{**c, "extra": tuple(c["extra"])}) for c in manifest["scenarios"]]
    require(cases and all(c.suite == "decode" for c in cases), "invalid decode scenarios")
    require(len({c.name for c in cases}) == len(cases), "duplicate decode scenario")
    require(len(summary["results"]) == len(cases), "incomplete decode summary")
    summaries = {r["scenario"]: r for r in summary["results"]}
    require(len(summaries) == len(cases) and set(summaries) == {c.name for c in cases}, "summary case mismatch")
    commands = manifest["commands"]
    require(len(commands) == 2 * len(cases), "incomplete phase commands")
    require(len({c["cache_salt"] for c in commands}) == len(commands), "phase cache salt reused")
    events = [json.loads(line) for line in text(directory + "events.jsonl").splitlines() if line]
    normalized_commands, rows = [], []
    for index, case in enumerate(cases):
        for warmup in (True, False):
            phase = MATRIX.phase_name(case, warmup)
            saved = commands[index * 2 + (0 if warmup else 1)]
            require(saved["scenario"] == case.name and saved["warmup"] is warmup, "phase command order mismatch")
            argv = saved["command"]
            require(argv[:3] == ["docker", "exec", args["container"]], "benchmark target mismatch")
            require(argv[3:6] == ["timeout", "--signal=TERM", "--kill-after=15s"]
                    and argv[7:10] == ["vllm", "bench", "serve"], "unexpected benchmark executable")
            for flag, value in (("--model", args["model"]), ("--served-model-name", args["served_model_name"])):
                require(option(argv, flag) == value, "benchmark model identity mismatch")
            require(float(option(argv, "--temperature")) == 0, "phase is not temperature zero")
            require(option(argv, "--num-warmups") == option(argv, "--ready-check-timeout-sec") == "0",
                    "hidden benchmark warmup/readiness requests")
            require("--ignore-eos" in argv and "--save-detailed" in argv, "missing fixed-length/detailed flags")
            for flag, value in (("--num-prompts", case.warmups if warmup else case.prompts),
                                ("--random-input-len", case.input_len), ("--random-output-len", case.output_len),
                                ("--max-concurrency", case.concurrency),
                                ("--seed", case.seed + (1_000_000 if warmup else 0))):
                require(int(option(argv, flag)) == value, f"scenario {flag} mismatch")
            salt = MATRIX.cache_salt(manifest["run_nonce"], case, warmup)
            require(saved["cache_salt"] == salt and json.loads(option(argv, "--extra-body")) == {"cache_salt": salt},
                    "missing/unexpected request body or cache salt")
            starts = [e for e in events if e.get("kind") == "command_start" and e.get("log") == phase + ".log"]
            ends = [e for e in events if e.get("kind") == "command_end" and e.get("log") == phase + ".log"]
            require(len(starts) == len(ends) == 1 and starts[0]["argv"] == argv
                    and ends[0]["returncode"] == 0, "actual command/exit evidence mismatch")
            normalized = argv[:]
            normalized[2] = "CONTAINER"
            for flag in ("--base-url", "--result-dir", "--extra-body"):
                normalized[normalized.index(flag) + 1] = "RUN_SPECIFIC"
            normalized_commands.append(normalized)
            result = read(directory + phase + ".json")
            row = MATRIX.result_row(result, case, warmup, targets["cases"][phase]["output_lens"])
            require(result["total_input_tokens"] == sum(result["input_lens"])
                    and result["total_output_tokens"] == sum(result["output_lens"]), "inconsistent token totals")
            for field, count in (("req_s", row["completed"]), ("out_tok_s", result["total_output_tokens"]),
                                 ("total_tok_s", result["total_input_tokens"] + result["total_output_tokens"])):
                require(math.isclose(row[field], count / row["duration_s"], rel_tol=1e-8), "inconsistent timing/rate totals")
            for field in ("ttfts", "latencies", "start_times", "queue_times"):
                require(all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in result[field]),
                        "invalid per-request timing")
            require(all(isinstance(values, list) and values
                        and all(type(x) in (int, float) and math.isfinite(x) and x >= 0 for x in values)
                        for values in result["itls"]), "invalid per-request ITLs")
            if not warmup:
                require(row == summaries[case.name], "summary differs from measured detailed result")
                before = mtp_series(text(directory + phase + ".metrics_before.txt"))
                after = mtp_series(text(directory + phase + ".metrics_after.txt"))
                rows.append({"scenario": case.name, "timing": row, "mtp": mtp_delta(before, after),
                             "input_lens": result["input_lens"], "output_lens": result["output_lens"],
                             "generated_text_sha256": [sha(s.encode()) for s in result["generated_texts"]]})
    return {"name": root.name, "config": config, "image": image, "commit": commit,
            "uncommitted_diff_sha256": diff_hash, "gpu": gpu, "server_command": command,
            "server_env": identity_env, "source_manifest": sources, "manifest": manifest,
            "expected_lengths": targets, "normalized_commands": normalized_commands,
            "rows": rows, "input_sha256": hashes}


def compare(base, candidate):
    for key in ("config", "gpu_util"):
        require(base["config"][key] == candidate["config"][key], f"different {key}")
    for key in ("image", "gpu", "server_command", "server_env", "normalized_commands", "expected_lengths"):
        require(base[key] == candidate[key], f"incompatible {key}")
    require(base["manifest"]["scenarios"] == candidate["manifest"]["scenarios"], "different scenarios/seeds")
    require(base["manifest"]["run_nonce"] != candidate["manifest"]["run_nonce"], "runs reused cache namespace")
    for field in ("model", "served_model_name", "temperature", "seed_offset"):
        require(base["manifest"]["arguments"][field] == candidate["manifest"]["arguments"][field], f"different {field}")
    common_sources = []
    for run in (base, candidate):
        extras = source_extras(run["config"]["variant"])
        common_sources.append({p: value for p, value in run["source_manifest"]["files"].items() if p not in extras})
    require(common_sources[0] == common_sources[1], "different non-MoE runtime sources")
    rows = []
    for a, b in zip(base["rows"], candidate["rows"], strict=True):
        require(a["scenario"] == b["scenario"] and a["input_lens"] == b["input_lens"]
                and a["output_lens"] == b["output_lens"], "different measured request lengths")
        change = {field: 100 * (b["timing"][field] / a["timing"][field] - 1) if a["timing"][field] else None
                  for field in ("out_tok_s", "total_tok_s", "med_ttft_ms", "med_itl_ms", "med_tpot_ms")}
        same = sum(x == y for x, y in zip(a["generated_text_sha256"], b["generated_text_sha256"], strict=True))
        rows.append({"scenario": a["scenario"], "baseline": a, "candidate": b,
                     "change_percent": change, "identical_generated_texts": same})
    prior_history_differs = base["config"]["suite"] != candidate["config"]["suite"]
    return {"status": "complete_compatible_pair", "baseline": base, "candidate": candidate, "rows": rows,
            "prior_suite_history_differs": prior_history_differs,
            "analysis_source_sha256": {p.name: sha(p.read_bytes()) for p in (Path(__file__), HERE / "serving_matrix.py")},
            "interpretation": "Explicit temperature zero controls sampling. MTP acceptance, generated paths, "
                              "startup autotuning and scheduling may still vary. This pair is not a confidence interval "
                              "or a model-quality pass. Different prior suites are recorded as a preconditioning limitation."}


def markdown(report):
    a, b = report["baseline"], report["candidate"]
    lines = [f"Baseline `{a['name']}`; candidate `{b['name']}`; explicit temperature 0; "
             f"GPU utilization {a['config']['gpu_util']}.", "",
             f"Prior suites: `{a['config']['suite']}` / `{b['config']['suite']}`. "
             "Each case has its own warmup and measured cache salt.", "",
             "| task | output tok/s base / candidate | change | TTFT p50 ms | ITL p50 ms | MTP acceptance | acceptance length | identical output texts |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["rows"]:
        x, y = row["baseline"], row["candidate"]
        t, u, m, n = x["timing"], y["timing"], x["mtp"], y["mtp"]
        lines.append(f"| {row['scenario']} | {t['out_tok_s']:.2f} / {u['out_tok_s']:.2f} | "
                     f"{row['change_percent']['out_tok_s']:+.2f}% | {t['med_ttft_ms']:.2f} / {u['med_ttft_ms']:.2f} | "
                     f"{t['med_itl_ms']:.3f} / {u['med_itl_ms']:.3f} | "
                     f"{100*m['acceptance_rate']:.2f}% / {100*n['acceptance_rate']:.2f}% | "
                     f"{m['acceptance_length']:.3f} / {n['acceptance_length']:.3f} | "
                     f"{row['identical_generated_texts']}/{t['completed']} |")
    lines += ["", "MTP acceptance = accepted/drafted tokens; acceptance length = 1 + accepted tokens/draft steps. "
              "Both use measured-phase counter deltas. Matching generated texts is diagnostic, not a quality score.", "",
              report["interpretation"]]
    if report["prior_suite_history_differs"]:
        lines += ["", "The prior suites differ: preconditioning is not identical despite matching per-case warmups. "
                  "Use this result as a diagnostic and repeat with matched run order/history before a causal timing claim."]
    return "\n".join(lines) + "\n"


def preflight():
    import copy
    import tempfile

    cases = MATRIX.scenarios("decode", 0)[:2]  # Deliberately not a fixed 4/17-row requirement.
    with tempfile.TemporaryDirectory() as directory:
        roots = [Path(directory) / name for name in ("baseline", "candidate")]
        for root, variant in zip(roots, ("next", "shared_exact")):
            root.mkdir()
            def save(name, value):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value if isinstance(value, str) else json.dumps(value))
            save("exit_code.txt", "0\n")
            save("config.txt", f"tag={root.name}\nvariant={variant}\nsuite=diagnostic\nconfig=next\ngpu_util=0.78\ngreedy_decode=1\nnll_rounds=5\n")
            save("image.txt", "sha256:fixture\n")
            save("commit.txt", "fixture\n")
            save("uncommitted.diff", "")
            save("gpu_before.csv", "name,uuid,driver_version,memory.total [MiB],power.limit [W]\nSM120,GPU-fixture,1,97887 MiB,600 W\n")
            command = ["serve", "fixture", "--speculative-config", '{"method":"mtp","num_speculative_tokens":2}']
            env = ["COMMON=1", "VLLM_CACHE_ROOT=" + variant]
            if variant != "next":
                command += ["--moe-backend", "flashinfer_b12x"]
                env += ["VLLM_B12X_FIX_SCALES=1", "VLLM_B12X_SHARED_WORKSPACE=1"]
            common = {p: b"fixture\n" for p in ("scripts/serve.sh", "scripts/cudagraph_sizes.txt",
                                                "bench/qwen_experiment.sh", "bench/serving_matrix.py")}
            contents = {**common, **{p: b"overlay\n" for p in source_extras(variant)}}
            for name, data in contents.items():
                save("source_files/" + name, data.decode())
            save("source_manifest.json", {"schema_version": 1, "variant": variant,
                                           "files": {p: sha(data) for p, data in contents.items()}})
            save("source_verify.log", {"status": "passed", "action": "verify", "files": len(contents), "variant": variant})
            save("container.json", [{"Name": "/qwen-exp-" + root.name, "Image": "sha256:fixture",
                                      "Config": {"Cmd": command, "Env": env}, "Mounts": []}])
            args = MATRIX.parser().parse_args(["--container", "qwen-exp-" + root.name, "--output-dir", str(root / "serving"),
                                               "--suite", "decode", "--temperature", "0"])
            prefix = "serving/serving_matrix_decode/"
            remote = "/bench-results/serving_matrix_decode"
            nonce = root.name
            commands, targets, summaries, events = [], {}, [], []
            for case in cases:
                for warmup in (True, False):
                    name = MATRIX.phase_name(case, warmup)
                    argv = MATRIX.bench_command(args, case, warmup, remote, nonce)
                    commands.append({"scenario": case.name, "warmup": warmup, "command": argv,
                                     "cache_salt": MATRIX.cache_salt(nonce, case, warmup)})
                    events += [{"kind": "command_start", "log": name + ".log", "argv": argv},
                               {"kind": "command_end", "log": name + ".log", "returncode": 0}]
                    count = case.warmups if warmup else case.prompts
                    result = {"num_prompts": count, "max_concurrency": case.concurrency, "completed": count,
                              "failed": 0, **{field: 1.0 for field in MATRIX.METRICS.values()},
                              **{field: [1.0] * count for field in MATRIX.DETAILS},
                              "input_lens": [128] * count, "output_lens": [512] * count,
                              "errors": [""] * count, "generated_texts": ["fixture"] * count,
                              "itls": [[0.001]] * count, "duration": 2.0,
                              "total_input_tokens": 128 * count, "total_output_tokens": 512 * count,
                              "request_throughput": count / 2, "output_throughput": 512 * count / 2,
                              "total_token_throughput": 640 * count / 2}
                    save(prefix + name + ".json", result)
                    targets[name] = {"output_lens": [512] * count}
                    if not warmup:
                        summaries.append(MATRIX.result_row(result, case, False))
                        for phase, values in (("before", (100, 200, 100)), ("after", (110, 220, 112))):
                            save(prefix + name + f".metrics_{phase}.txt", "\n".join(
                                f'vllm:spec_decode_{metric}_total{{model_name="bench"}} {value}'
                                for metric, value in zip(COUNTERS, values)))
            save(prefix + "manifest.json", {"arguments": vars(args), "run_nonce": nonce,
                                            "scenarios": [asdict(c) for c in cases], "commands": commands})
            save(prefix + "expected_output_lengths.json", {"cases": targets, "sampler": None})
            save(prefix + "summary.json", {"status": "passed", "results": summaries, "error": None})
            save(prefix + "events.jsonl", "\n".join(json.dumps(e) for e in events))
        a, b = map(load, roots)
        report = compare(a, b)
        require(len(report["rows"]) == 2 and all(r["change_percent"]["out_tok_s"] == 0 for r in report["rows"]),
                "valid variable-sized pair failed")
        require(report["rows"][0]["baseline"]["mtp"]["acceptance_rate"] == 0.6, "MTP arithmetic error")
        require("0.00%" in markdown(report), "Markdown table failed")
        rejected = 0
        def sampled_phase(manifest):
            command = manifest["commands"][1]["command"]
            command[command.index("--temperature") + 1] = "1"
            return manifest
        mutations = [
            ("serving/serving_matrix_decode/manifest.json", sampled_phase),
            ("exit_code.txt", lambda x: "1"),
            ("source_files/bench/serving_matrix.py", lambda x: "changed"),
            ("serving/serving_matrix_decode/summary.json", lambda x: {**x, "results": x["results"][:1]}),
            ("serving/serving_matrix_decode/decode-c1.measured.json", lambda x: {**x, "output_lens": [1] * len(x["output_lens"])}),
            ("serving/serving_matrix_decode/decode-c1.measured.metrics_before.txt", lambda x: ""),
            ("serving/serving_matrix_decode/decode-c1.measured.metrics_after.txt", lambda x: x.replace("110", "90")),
        ]
        for name, mutate in mutations:
            path = roots[1] / name
            previous = path.read_text()
            data = json.loads(previous) if name.endswith(".json") else previous
            changed = mutate(data)
            path.write_text(json.dumps(changed) if name.endswith(".json") else changed)
            try:
                load(roots[1])
            except ValueError:
                rejected += 1
            else:
                raise AssertionError("malformed run accepted")
            finally:
                path.write_text(previous)
        for mutate in (lambda x: x.update(image="different"),
                       lambda x: x["gpu"].update(uuid="different"),
                       lambda x: x["manifest"]["arguments"].update(temperature=1),
                       lambda x: x["source_manifest"]["files"].update({"bench/serving_matrix.py": "different"}),
                       lambda x: x["manifest"].update(run_nonce=a["manifest"]["run_nonce"]),
                       lambda x: x["rows"][0]["input_lens"].__setitem__(0, 129)):
            broken = copy.deepcopy(b)
            mutate(broken)
            try:
                compare(a, broken)
            except ValueError:
                rejected += 1
            else:
                raise AssertionError("incompatible pair accepted")
    print(json.dumps({"status": "passed", "mode": "cpu_preflight", "http_requests": 0, "gpu_work": False,
                      "variable_scenario_count": 2, "malformed_fixtures_rejected": rejected,
                      "script_sha256": sha(Path(__file__).read_bytes())}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", nargs="?", type=Path)
    parser.add_argument("candidate", nargs="?", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        require(args.baseline is None and args.candidate is None, "preflight does not accept run directories")
        preflight()
        return
    require(args.baseline is not None and args.candidate is not None, "baseline and candidate are required")
    report = compare(load(args.baseline), load(args.candidate))
    content = markdown(report)
    if args.output_prefix is not None:
        args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        Path(str(args.output_prefix) + ".json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        Path(str(args.output_prefix) + ".md").write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
