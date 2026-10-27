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
IDENTITY_SPEC = importlib.util.spec_from_file_location("paired_runtime_identity", HERE / "compare_runtime_identity.py")
IDENTITY = importlib.util.module_from_spec(IDENTITY_SPEC)
IDENTITY_SPEC.loader.exec_module(IDENTITY)
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
    return IDENTITY.source_extras(variant)


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
    runtime = IDENTITY.load(root, text, read)
    config = runtime["config"]
    require(config["greedy_decode"] == "1", "invalid greedy run identity")
    container_name = "qwen-exp-" + config["tag"]
    directory = "serving/serving_matrix_decode/"
    manifest, summary = read(directory + "manifest.json"), read(directory + "summary.json")
    targets = read(directory + "expected_output_lengths.json")
    args = manifest["arguments"]
    require(args["suite"] == "decode" and args.get("temperature") == 0, "not explicit greedy decode")
    require(args["container"] == container_name, "request/container identity mismatch")
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
    return {**runtime, "manifest": manifest,
            "expected_lengths": targets, "normalized_commands": normalized_commands,
            "rows": rows, "input_sha256": hashes}


def compare(base, candidate):
    runtime_changes = IDENTITY.compare(base, candidate)
    for key in ("normalized_commands", "expected_lengths"):
        require(base[key] == candidate[key], f"incompatible {key}")
    require(base["manifest"]["scenarios"] == candidate["manifest"]["scenarios"], "different scenarios/seeds")
    require(base["manifest"]["run_nonce"] != candidate["manifest"]["run_nonce"], "runs reused cache namespace")
    for field in ("model", "served_model_name", "temperature", "seed_offset"):
        require(base["manifest"]["arguments"][field] == candidate["manifest"]["arguments"][field], f"different {field}")
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
            "prior_suite_history_differs": prior_history_differs, "runtime_changes": runtime_changes,
            "analysis_source_sha256": {p.name: sha(p.read_bytes()) for p in (Path(__file__), HERE / "serving_matrix.py", HERE / "compare_runtime_identity.py")},
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
              report["interpretation"], "", "Validated runtime changes: " + json.dumps(report["runtime_changes"], sort_keys=True)]
    if report["prior_suite_history_differs"]:
        lines += ["", "The prior suites differ: preconditioning is not identical despite matching per-case warmups. "
                  "Use this result as a diagnostic and repeat with matched run order/history before a causal timing claim."]
    return "\n".join(lines) + "\n"


def preflight():
    require("kernels/moe-nvfp4/shared_exact/vllm/" + EXPERT in source_extras("shared_exact_gdn"),
            "composed GDN variant must retain the original MoE parent source")
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
            engine = "Initializing a V1 LLM engine (v0.32.0) with config: model='fixture', compilation_config={'splitting_ops': " + repr(IDENTITY.STANDARD_SPLITS+IDENTITY.AUTO_SPLITS) + ", 'pass_config': {'fuse_rope_kvcache_cat_mla': False}}, kernel_config=KernelConfig(moe_backend='" + ("auto" if variant == "next" else "flashinfer_b12x") + "')"
            save("startup.log", engine)
            env = ["COMMON=1", "VLLM_CACHE_ROOT=/root/.cache/vllm/qwen-sequential-" + variant]
            if variant != "next":
                command += ["--moe-backend", "flashinfer_b12x"]
                env += ["VLLM_B12X_FIX_SCALES=1", "VLLM_B12X_SHARED_WORKSPACE=1"]
            common = {p: b"fixture\n" for p in ("scripts/serve.sh", "scripts/cudagraph_sizes.txt",
                                                "bench/qwen_experiment.sh", "bench/serving_matrix.py", "bench/source_manifest.py")}
            contents = {**common, **{p: b"overlay\n" for p in source_extras(variant)},
                        **{p: b"mounted\n" for p in IDENTITY.expected_mounts(variant).values()}}
            for name, data in contents.items():
                save("source_files/" + name, data.decode())
            save("source_manifest.json", {"schema_version": 1, "variant": variant,
                                           "files": {p: sha(data) for p, data in contents.items()}})
            save("source_verify.log", {"status": "passed", "action": "verify", "files": len(contents), "variant": variant})
            save("container.json", [{"Name": "/qwen-exp-" + root.name, "Image": "sha256:fixture",
                                      "Config": {"Cmd": command, "Env": env}, "Mounts": [
                {"Type":"bind", "Source":"/fixture/" + src, "Destination":dst, "RW":False, "Mode":"ro"}
                for dst,src in IDENTITY.expected_mounts(variant).items()] + [
                {"Type":"bind", "Source":"/fixture/results/qwen_sequential/" + root.name + "/serving" if dst == "/bench-results" else "/fixture" + dst,
                 "Destination":dst,"RW":dst != "/root/.cache/huggingface","Mode":"ro" if dst == "/root/.cache/huggingface" else ""}
                for dst in ("/bench-results","/root/.cache/vllm","/root/.cache/flashinfer","/root/.cache/huggingface")]}])
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
        # Complete composed-run fixtures exercise disk evidence, not normalized objects.
        root = roots[1]
        def save_gdn(name, value):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(value if isinstance(value, str) else json.dumps(value))
        for variant in ("shared_exact_gdn", "scalar_gdn"):
            save_gdn("config.txt", f"tag={root.name}\nvariant={variant}\nsuite=diagnostic\nconfig=next\ngpu_util=0.78\ngreedy_decode=1\nnll_rounds=5\n")
            container = json.loads((root / "container.json").read_text())[0]
            command = container["Config"]["Cmd"]
            if "--compilation-config" in command:
                i = command.index("--compilation-config"); del command[i:i+2]
            compilation = {"splitting_ops": IDENTITY.STANDARD_SPLITS + IDENTITY.AUTO_SPLITS + [IDENTITY.GDN_OP],
                           "pass_config":{"fuse_rope_kvcache_cat_mla":False}}
            command += ["--compilation-config", json.dumps(compilation)]
            env = {k:v for k,v in (x.split("=",1) for x in container["Config"]["Env"])}
            env.update(VLLM_CACHE_ROOT="/root/.cache/vllm/qwen-sequential-" + variant,
                       VLLM_GDN_NORM_FP8_FUSION="1", VLLM_B12X_SCALAR_INPUT_SCALE="1" if variant == "scalar_gdn" else "0")
            container["Config"]["Env"] = [k+"="+v for k,v in env.items()]
            container["Mounts"] = [m for m in container["Mounts"] if not m["Destination"].startswith(PKG)] + [
                {"Type":"bind", "Source":"/fixture/"+src, "Destination":dst, "RW":False, "Mode":"ro"}
                for dst,src in IDENTITY.expected_mounts(variant).items()]
            save_gdn("container.json", [container])
            sources = {p: v for p,v in b["source_manifest"]["files"].items()
                       if p not in source_extras("shared_exact") and p != "build/vllm/"+IDENTITY.GDN+IDENTITY.GDN_FILES[0]}
            for name in source_extras(variant) | set(IDENTITY.expected_mounts(variant).values()):
                value = "mounted\n" if name in IDENTITY.expected_mounts(variant).values() else "helper\n"
                save_gdn("source_files/"+name, value); sources[name] = sha(value.encode())
            save_gdn("source_manifest.json", {"schema_version":1,"variant":variant,"files":sources})
            save_gdn("source_verify.log", {"status":"passed","action":"verify","variant":variant,"files":len(sources)})
            save_gdn("compilation_config.json", compilation)
            layers = [i for i in range(40) if (i+1)%4]
            startup = "Initializing a V1 LLM engine (v0.32.0) with config: model='fixture', compilation_config={'splitting_ops': " + repr(compilation["splitting_ops"]) + ", 'pass_config': {'fuse_rope_kvcache_cat_mla': False}}, kernel_config=KernelConfig(moe_backend='flashinfer_b12x')\n"
            startup += "\n".join(f"GDN norm-FP8 ready for model.layers.{i}.linear_attn: compiled fallback" for i in layers)
            save_gdn("startup.log", startup)
            save_gdn("gdn_readiness.json", {"ready_layers":layers,"expected_layers":layers,"disabled":[]})
            composed = load(root)
            paired = compare(a, composed)
            require(paired["runtime_changes"]["candidate_moe_parent"] == variant.removesuffix("_gdn"), "wrong MoE parent reported")
            require(composed["gdn_enabled"] and len(composed["gdn_readiness"]["ready_layers"]) == 30, "GDN readiness not checked")
            malformed = [
                ("container.json", lambda x: [{**x[0],"Mounts":[m for m in x[0]["Mounts"] if not m["Destination"].endswith("gdn_norm_fp8_quant_out.py")]}]),
                ("container.json", lambda x: [{**x[0],"Mounts":[{**m,"RW":True} if m["Destination"].endswith("gdn_norm_fp8_kernel.py") else m for m in x[0]["Mounts"]]}]),
                ("container.json", lambda x: [{**x[0],"Mounts":[{**m,"Source":"/fixture/kernels/moe-nvfp4/shared/vllm/"+EXPERT} if m["Destination"]==PKG+EXPERT else m for m in x[0]["Mounts"]]}]),
                ("container.json", lambda x: [{**x[0],"Config":{**x[0]["Config"],"Env":[e for e in x[0]["Config"]["Env"] if not e.startswith("VLLM_GDN_NORM_FP8_FUSION=")]}}]),
                ("compilation_config.json", lambda x: {"splitting_ops":x["splitting_ops"][:-1]}),
                ("compilation_config.json", lambda x: {**x,"mode":0}),
                ("gdn_readiness.json", lambda x: {**x,"ready_layers":x["ready_layers"][:-1]}),
                ("gdn_readiness.json", lambda x: {**x,"ready_layers":x["ready_layers"]+[0]}),
                ("startup.log", lambda x: x.replace("'fuse_rope_kvcache_cat_mla': False", "'fuse_rope_kvcache_cat_mla': True")),
                ("startup.log", lambda x: x+"\nGDN norm-FP8 disabled for model.layers.0.linear_attn"),
                ("startup.log", lambda x: x.replace("vllm::"+"gdn_attention_core_norm_fp8_packed_experimental", "vllm::wrong")),
                ("source_files/"+IDENTITY.GDN_ROOT+"gdn_norm_fp8_quant_out.py", lambda x:x+"changed"),
            ]
            for name,mutate in malformed:
                path=root/name; before=path.read_text(); data=json.loads(before) if name.endswith(".json") else before
                changed=mutate(data); path.write_text(json.dumps(changed) if name.endswith(".json") else changed)
                try:
                    load(root)
                except ValueError:
                    rejected += 1
                else:
                    raise AssertionError("malformed GDN evidence accepted: "+name)
                finally:
                    path.write_text(before)
            for key in ("server_env", "server_command"):
                broken=copy.deepcopy(composed)
                if key=="server_env":broken[key]["UNRELATED_FLAG"]="1"
                else:broken[key].append("--unrelated-override")
                try:compare(a,broken)
                except ValueError:rejected+=1
                else:raise AssertionError("unrelated runtime override accepted")
            parent_copy=copy.deepcopy(composed)
            parent_copy["manifest"]["run_nonce"]="other"
            expert="kernels/moe-nvfp4/"+variant.removesuffix("_gdn")+"/vllm/"+EXPERT
            parent_copy["source_manifest"]["files"][expert]="changed-but-same-parent"
            try:compare(composed,parent_copy)
            except ValueError:rejected+=1
            else:raise AssertionError("same-parent source drift accepted")
        # Matched-FI fixture uses the frozen real reference and verifier sources,
        # with synthetic worker evidence (no vLLM imports or device initialization).
        local = HERE.parent / IDENTITY.FI_ROOT
        config_path = root / "config.txt"
        config_path.write_text(config_path.read_text()+"fi_control=1\nfi_source_sha256="+IDENTITY.FI_SHA+"\n")
        sources=json.loads((root/"source_manifest.json").read_text())
        sources["fi_control"]=True
        for relative in IDENTITY.FI_FILES:
            content=(local/relative).read_text()
            save_gdn("source_files/"+IDENTITY.FI_ROOT+relative,content)
            sources["files"][IDENTITY.FI_ROOT+relative]=sha(content.encode())
        save_gdn("source_manifest.json",sources)
        save_gdn("source_verify.log",{"status":"passed","action":"verify","variant":"scalar_gdn","files":len(sources["files"])})
        container=json.loads((root/"container.json").read_text())[0]
        container["Config"]["Env"] += [k+"="+v for k,v in IDENTITY.FI_ENV.items()]
        container["Config"]["Cmd"] += ["--shutdown-timeout", "30"]
        container["Mounts"].append({"Type":"bind","Source":"/fixture/kernels/decode-route-capture","Destination":"/opt/qwen-fi-control","RW":False,"Mode":"ro"})
        save_gdn("container.json",[container])
        document=json.loads((local/"references/next_cpu_audit_autotune.json").read_text())
        provenance=json.loads((local/"references/next_cpu_audit_autotune.provenance.json").read_text())
        table_hash=provenance["reference_entries_sha256"]
        prefix="serving/fi_control/autotune/"
        save_gdn(prefix+"source.json",(local/"references/next_cpu_audit_autotune.json").read_text())
        stages=("preloaded","loaded","after_warmup","server_startup","server_shutdown")
        verification={"server_checkpoint":"shutdown_verified","source_sha256":IDENTITY.FI_SHA,
            "import_mode":"fixed_reference_replay_v1",
            "preload":{"path":"/bench-results/fi_control/autotune/source.json","returned":True,"boundary":"before_worker_load_model"},
            "server_startup_boundary":"after_compile_or_warm_up_model",
            "reference_entries_sha256":table_hash,"reference_entries":132,
            "status":"reference_tactics_verified_without_lookup_observation",
            "compiled_graph_cache_imported":False,"lookup_observation_enabled":False,"serving_lookup_observer":False,
            "runtime_source_sha256":{"flashinfer_autotuner":"a"*64,"vllm_kernel_warmup":"b"*64},
            "load_calls":[{"path":"/bench-results/fi_control/cache/autotune_import/autotune_configs_dp0_rank0.json","returned":True}],
            "selected_inductor_config_count":1,
            "checks":{stage:{"reference_entries":132,"actual_entries":132,"entries_sha256":table_hash,
                             "missing":[],"extra":[],"changed":[]} for stage in stages}}
        save_gdn(prefix+"verification.json",verification)
        for stage in stages:
            save_gdn(prefix+stage+"_tactics.json",{k:v for k,v in document.items() if k.startswith("(")})
        for stage in ("startup","shutdown"):
            save_gdn(prefix+"server_"+stage+"_effective.json",document)
            save_gdn("fi_"+stage+"_verify.json",{"verified_stage":"server_"+stage,"source_sha256":IDENTITY.FI_SHA,
                    "effective_entries_sha256":table_hash,"entries":132,"inductor_pinned":False,"serving_lookup_observer":False,
                    "import_mode":"fixed_reference_replay_v1","preload_boundary":"before_worker_load_model"})
        selected={"XBLOCK":1,"R0_BLOCK":1024,"num_warps":8}
        save_gdn(prefix+"compiler_config_audit.json",{"files":[{"path":"fixture.best_config","selected":selected,
                    "selected_sha256":sha(json.dumps(selected,sort_keys=True,separators=(",",":")).encode())}]})
        controlled=load(root)
        other=copy.deepcopy(controlled);other["manifest"]["run_nonce"]="other-controlled"
        require(compare(controlled,other)["runtime_changes"]["fi_control"],"matched FI identity not reported")
        # Changed/new/removed compiler kernels are observable differences, not
        # a reason to reject a composed graph with matched FI reference tactics.
        different=copy.deepcopy(other)
        audit=different["fi_evidence"]["compiler_config_audit"]
        audit["files"][0]["selected"]={"XBLOCK":2}
        audit["files"][0]["selected_sha256"]=sha(b'{"XBLOCK":2}')
        audit["files"].append({"path":"new_gdn.best_config","selected":{},"selected_sha256":sha(b'{}')})
        changes=compare(controlled,different)["runtime_changes"]["inductor_config_changes"]
        require("fixture.best_config" in changes["changed"] and "new_gdn.best_config" in changes["added"],
                "Inductor selection/new-kernel changes were not reported")
        reverse=compare(different,controlled)["runtime_changes"]["inductor_config_changes"]
        require("new_gdn.best_config" in reverse["removed"],"removed Inductor kernel not reported")
        try:compare(a,controlled)
        except ValueError:rejected+=1
        else:raise AssertionError("controlled-vs-unpinned pair accepted")
        def altered_effective(x):
            key=next(k for k in x if k.startswith("("));x[key]=[x[key][0],-123];return x
        def shutdown_timeout(x, value, extra=()):
            argv=x[0]["Config"]["Cmd"]; index=argv.index("--shutdown-timeout")
            if value is None: del argv[index:index+2]
            else: argv[index+1]=value
            argv.extend(extra)
            return x
        malformed=[
            ("container.json",lambda x:shutdown_timeout(x,None)),
            ("container.json",lambda x:shutdown_timeout(x,"0")),
            ("container.json",lambda x:shutdown_timeout(x,"-1")),
            ("container.json",lambda x:shutdown_timeout(x,"29")),
            ("container.json",lambda x:shutdown_timeout(x,"invalid")),
            ("container.json",lambda x:shutdown_timeout(x,"30",("--shutdown-timeout","30"))),
            ("container.json",lambda x:shutdown_timeout(x,"30",("--shutdown-timeout=0",))),
            ("config.txt",lambda x:x.replace("fi_source_sha256="+IDENTITY.FI_SHA,"fi_source_sha256="+"0"*64)),
            ("container.json",lambda x:[{**x[0],"Mounts":[{**m,"RW":True} if m["Destination"]=="/opt/qwen-fi-control" else m for m in x[0]["Mounts"]]}]),
            (prefix+"source.json",altered_effective),
            (prefix+"preloaded_tactics.json",altered_effective),
            (prefix+"verification.json",lambda x:{**x,"import_mode":"legacy_file_load"}),
            (prefix+"verification.json",lambda x:{**x,"preload":{**x["preload"],"returned":False}}),
            (prefix+"verification.json",lambda x:{**x,"server_startup_boundary":"before_graph_capture"}),
            (prefix+"server_startup_effective.json",altered_effective),
            (prefix+"server_shutdown_effective.json",altered_effective),
            (prefix+"verification.json",lambda x:{**x,"server_checkpoint":"warmup_verified"}),
            (prefix+"verification.json",lambda x:{**x,"reference_entries_sha256":"0"*64}),
            (prefix+"verification.json",lambda x:{**x,"load_calls":[{"returned":False}]}),
            (prefix+"verification.json",lambda x:{**x,"checks":{k:v for k,v in x["checks"].items() if k!="loaded"}}),
            ("fi_startup_verify.json",lambda x:{**x,"entries":131}),
            ("fi_shutdown_verify.json",lambda x:{**x,"effective_entries_sha256":"0"*64}),
            (prefix+"compiler_config_audit.json",lambda x:{"files":[{**x["files"][0],"selected":{"XBLOCK":99}}]}),
        ]
        for name,mutate in malformed:
            path=root/name;before=path.read_text();value=json.loads(before) if name.endswith(".json") else before
            changed=mutate(value);path.write_text(json.dumps(changed) if name.endswith(".json") else changed)
            try:load(root)
            except ValueError:rejected+=1
            else:raise AssertionError("malformed FI evidence accepted: "+name)
            finally:path.write_text(before)
    print(json.dumps({"status": "passed", "mode": "cpu_preflight", "http_requests": 0, "gpu_work": False,
                      "variable_scenario_count": 2, "malformed_fixtures_rejected": rejected,
                      "fi_shutdown_timeout_seconds": 30, "shutdown_negative_fixtures": 7,
                      "script_sha256": sha(Path(__file__).read_bytes()),
                      "identity_sha256": sha((HERE / "compare_runtime_identity.py").read_bytes())}))


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
