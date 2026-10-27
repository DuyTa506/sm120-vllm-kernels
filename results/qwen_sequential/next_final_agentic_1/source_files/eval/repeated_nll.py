"""Repeated cold prompt NLL diagnostic; requests are explicit, preflight/compare are offline.

BENCH_URL=http://127.0.0.1:8132 python3 -I eval/repeated_nll.py --output run.json --rounds 5
python3 -I eval/repeated_nll.py --preflight
python3 -I eval/repeated_nll.py --compare baseline.json candidate.json

The ten prompts exactly reproduce long_eval's six NLL cases and
verify_logprobs' four prompts. This is a diagnostic, not a model-quality gate.
"""
import argparse
import ast
from datetime import datetime, timezone
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
REQUEST = {"model": "bench", "temperature": 0, "prompt_logprobs": 0, "max_tokens": 1}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive_int(value):
    return type(value) is int and value > 0


def corpus(paths, max_chars=None, separator="\n\n"):
    require(bool(paths), "empty corpus file list")
    parts, sources = [], {}
    for path in paths:
        data = Path(path).read_bytes()
        # open(..., errors='ignore') in the original probes uses universal newlines.
        with open(path, errors="ignore") as stream:
            parts.append(stream.read())
        sources[str(path)] = sha(data)
        if max_chars is not None and sum(map(len, parts)) > max_chars:
            break
    text = separator.join(parts)
    if max_chars is not None:
        require(len(text) >= max_chars, f"corpus shorter than {max_chars} characters")
        text = text[:max_chars]
    return text, sources


def build_prompts():
    docs = {
        "licenses": sorted(glob.glob("/usr/share/common-licenses/*")),
        "py_stdlib": sorted(glob.glob("/usr/lib/python3*/asyncio/*.py"))[:40],
        "py_email": sorted(glob.glob("/usr/lib/python3*/email/*.py"))[:40],
    }
    prompts = []
    for name, paths in docs.items():
        for chars in (30000, 60000):
            text, sources = corpus(paths, chars)
            prompts.append({"id": f"{name}-{chars}", "group": "long_nll", "text": text,
                            "characters": chars, "source_sha256": sources})
    paths = sorted(glob.glob("/usr/lib/python3*/json/*.py")) + sorted(glob.glob("/usr/lib/python3*/http/*.py"))
    text, sources = corpus(paths, separator="")
    for i, (chars, off) in enumerate(((2400, 0), (2700, 5000), (14000, 20000), (30000, 40000))):
        piece = text[off:off + chars]
        require(len(piece) == chars, f"short logprob corpus at offset {off}")
        prompt = f"[sample {i}]\n" + piece
        prompts.append({"id": f"logprobs-{i}", "group": "logprobs", "text": prompt,
                        "characters": len(prompt), "offset": off, "source_sha256": sources})
    for prompt in prompts:
        prompt["prompt_sha256"] = sha(prompt["text"].encode())
    require(len(prompts) == 10, "expected exactly ten prompts")
    return prompts


def descriptors(prompts):
    return [{k: v for k, v in p.items() if k != "text"} for p in prompts]


def schedule(prompts, rounds, nonce):
    result = []
    for number in range(rounds):
        offset = number % len(prompts)
        for p in prompts[offset:] + prompts[:offset]:
            result.append({"round": number, "prompt_id": p["id"],
                           "cache_salt": f"repeated-nll:{nonce}:{number}:{p['id']}"})
    return result


def validate_response(response):
    require(isinstance(response, dict) and not response.get("error"), "invalid/error response")
    require(isinstance(response.get("id"), str) and bool(response["id"]), "missing response id")
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "expected exactly one choice")
    choice = choices[0]
    require(isinstance(choice, dict) and type(choice.get("index")) is int and choice["index"] == 0,
            "invalid choice index")
    require(choice.get("finish_reason") in ("length", "stop"), "unfinished/failed completion")
    require(isinstance(choice.get("text"), str), "missing completion text")
    usage = response.get("usage")
    require(isinstance(usage, dict), "missing usage")
    count = usage.get("prompt_tokens")
    require(positive_int(count) and count > 1, "invalid prompt token count")
    require(type(usage.get("completion_tokens")) is int and usage["completion_tokens"] == 1,
            "expected exactly one generated token")
    require(type(usage.get("total_tokens")) is int and usage["total_tokens"] == count + 1,
            "inconsistent total token count")
    entries = choice.get("prompt_logprobs")
    require(isinstance(entries, list) and len(entries) == count, "prompt logprob length mismatch")
    require(entries[0] is None, "first prompt token must have no logprob")
    values, token_ids = [], []
    for entry in entries[1:]:
        require(isinstance(entry, dict) and len(entry) == 1, "missing/extra prompt logprob entries")
        token, item = next(iter(entry.items()))
        require(isinstance(token, str) and token.isascii() and token.isdigit(), "invalid token id")
        require(isinstance(item, dict), "invalid logprob record")
        value = item.get("logprob")
        require(type(value) in (int, float) and math.isfinite(value) and value <= 0,
                "nonfinite/positive/non-numeric logprob")
        values.append(float(value))
        token_ids.append(int(token))
    n = len(values)
    return {"prompt_tokens": count, "scored_tokens": n, "completion_tokens": 1,
            "nll": -statistics.mean(values),
            "nll_last_half": -statistics.mean(values[n // 2:]),
            "logprobs": values, "token_ids_sha256": sha(json.dumps(token_ids).encode()),
            "response_id": response["id"], "finish_reason": choice["finish_reason"],
            "generated_text": choice["text"]}


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".",
                                         suffix=".tmp", delete=False) as stream:
            temp = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp is not None and temp.exists():
            temp.unlink()


def summarize(record):
    require(record.get("status") == "passed", "run did not finish successfully")
    prompts, rounds = record["prompts"], record["rounds"]
    require(positive_int(rounds), "invalid rounds")
    require(len(prompts) == 10 and len({p["id"] for p in prompts}) == 10, "invalid prompt manifest")
    planned = schedule(prompts, rounds, record["run_nonce"])
    require(record["schedule"] == planned, "request schedule mismatch")
    rows = record["results"]
    require(len(rows) == len(planned), "incomplete request count")
    require(len({r["cache_salt"] for r in rows}) == len(rows), "cache salt reused")
    for row, expected in zip(rows, planned, strict=True):
        require(all(row[k] == v for k, v in expected.items()), "request order/id mismatch")
        require(row["completion_tokens"] == 1 and row["finish_reason"] in ("length", "stop"),
                "invalid stored completion")
        values = row["logprobs"]
        require(row["scored_tokens"] == len(values) == row["prompt_tokens"] - 1 and bool(values),
                "stored logprob length mismatch")
        require(all(type(x) in (int, float) and math.isfinite(x) and x <= 0 for x in values),
                "invalid stored logprobs")
        require(row["nll"] == -statistics.mean(values), "stored NLL mismatch")
        require(row["nll_last_half"] == -statistics.mean(values[len(values) // 2:]),
                "stored last-half NLL mismatch")
    summary = []
    for p in prompts:
        group = [r for r in rows if r["prompt_id"] == p["id"]]
        require(len(group) == rounds, "incomplete prompt repeats")
        require(len({(r["prompt_tokens"], r["token_ids_sha256"]) for r in group}) == 1,
                "prompt tokenization changed across repeats")
        values = [r["nll"] for r in group]
        summary.append({"prompt_id": p["id"], "prompt_tokens": group[0]["prompt_tokens"],
                        "token_ids_sha256": group[0]["token_ids_sha256"], "rounds": rounds,
                        "nll": values, "mean": statistics.mean(values), "min": min(values),
                        "max": max(values), "range": max(values) - min(values),
                        "sample_stdev": statistics.stdev(values) if rounds > 1 else None})
    return summary


def compare(base, candidate):
    for field in ("schema_version", "prompts", "rounds", "request_settings", "probe_source_sha256"):
        require(base[field] == candidate[field], f"incompatible {field}")
    require(base["run_nonce"] != candidate["run_nonce"], "comparison reused the same run nonce")
    a, b = summarize(base), summarize(candidate)
    result = []
    for x, y in zip(a, b, strict=True):
        require((x["prompt_id"], x["prompt_tokens"], x["token_ids_sha256"]) ==
                (y["prompt_id"], y["prompt_tokens"], y["token_ids_sha256"]), "different tokenization")
        delta, spread = y["mean"] - x["mean"], max(x["range"], y["range"])
        result.append({"prompt_id": x["prompt_id"], "baseline": x, "candidate": y,
                       "mean_delta": delta, "larger_within_backend_range": spread,
                       "delta_over_larger_range": delta / spread if spread else None})
    return {"status": "diagnostic_only", "rows": result,
            "interpretation": "Positive delta means higher NLL. Ranges describe observed repeats; "
                              "they are not confidence bounds or a model-quality acceptance threshold."}


def run(args):
    require(not args.output.exists(), f"output already exists: {args.output}")
    prompts = build_prompts()
    nonce = uuid.uuid4().hex
    sources = [Path(__file__).resolve(), ROOT / "eval/long_eval.py",
               ROOT / "kernels/moe-nvfp4/verify_logprobs.py"]
    record = {"schema_version": 1, "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
              "base_url": args.base_url, "rounds": args.rounds, "run_nonce": nonce,
              "request_settings": dict(REQUEST), "prompts": descriptors(prompts),
              "probe_source_sha256": {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in sources},
              "schedule": schedule(prompts, args.rounds, nonce), "results": []}
    atomic_save(args.output, record)
    by_id = {p["id"]: p for p in prompts}
    seen_tokens = {}
    try:
        for case in record["schedule"]:
            prompt = by_id[case["prompt_id"]]
            payload = {**REQUEST, "prompt": prompt["text"], "cache_salt": case["cache_salt"]}
            request = urllib.request.Request(args.base_url.rstrip("/") + "/v1/completions",
                                             json.dumps(payload).encode(), {"Content-Type": "application/json"})
            start = time.monotonic()
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                require(response.status == 200, f"HTTP status {response.status}")
                row = validate_response(json.load(response))
            tokenization = (row["prompt_tokens"], row["token_ids_sha256"])
            require(seen_tokens.setdefault(prompt["id"], tokenization) == tokenization,
                    "prompt tokenization changed across repeats")
            row.update(case, elapsed_s=time.monotonic() - start)
            record["results"].append(row)
            atomic_save(args.output, record)
            print(f"round={case['round']} prompt={prompt['id']} tokens={row['prompt_tokens']} "
                  f"nll={row['nll']:.6f}", flush=True)
        record["status"] = "passed"
        record["summary"] = summarize(record)
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        atomic_save(args.output, record)
        print(json.dumps({"status": "passed", "requests": len(record["results"]),
                          "interpretation": "Diagnostic collected; no model-quality pass is asserted."}))
    except BaseException as error:
        record["status"] = "failed"
        record["failure"] = {"type": type(error).__name__, "message": str(error),
                             "next_request_index": len(record["results"])}
        atomic_save(args.output, record)
        raise


def preflight(rounds):
    import copy
    import io
    from contextlib import redirect_stdout
    from types import SimpleNamespace
    from unittest.mock import patch

    prompts = build_prompts()
    # Execute only the pure corpus functions from the existing probes, not their HTTP loops.
    def original(path, name):
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        namespace = {"glob": glob}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        return namespace[name]
    doc_text = original(ROOT / "eval/long_eval.py", "doc_text")
    short_text = original(ROOT / "kernels/moe-nvfp4/verify_logprobs.py", "text")
    for i, p in enumerate(prompts):
        if p["group"] == "long_nll":
            expected = doc_text(list(p["source_sha256"]), p["characters"])
        else:
            number = i - 6
            expected = f"[sample {number}]\n" + short_text(p["characters"] - len(f"[sample {number}]\n"), p["offset"])
        require(p["text"] == expected, "prompt changed from the original probe")
    planned = schedule(prompts, rounds, "preflight")
    require(len({r["cache_salt"] for r in planned}) == 10 * rounds, "salts collide")
    valid = {"id": "synthetic", "choices": [{"index": 0, "text": "x", "finish_reason": "length",
             "prompt_logprobs": [None, {"11": {"logprob": -1.0}}, {"12": {"logprob": -2.0}}]}],
             "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}
    row = validate_response(valid)
    rejected = 0
    mutations = [
        lambda x: x.pop("usage"),
        lambda x: x["choices"].append(copy.deepcopy(x["choices"][0])),
        lambda x: x["choices"][0].update(finish_reason="error"),
        lambda x: x["usage"].update(completion_tokens=0),
        lambda x: x["usage"].update(total_tokens=9),
        lambda x: x["usage"].update(prompt_tokens=True),
        lambda x: x["choices"][0]["prompt_logprobs"].pop(),
        lambda x: x["choices"][0]["prompt_logprobs"].__setitem__(1, None),
        lambda x: x["choices"][0]["prompt_logprobs"].__setitem__(1, {"bad": {"logprob": -1}}),
        lambda x: x["choices"][0]["prompt_logprobs"][1]["11"].update(logprob=float("nan")),
        lambda x: x["choices"][0]["prompt_logprobs"][1]["11"].update(logprob=float("inf")),
        lambda x: x["choices"][0]["prompt_logprobs"][1]["11"].update(logprob=0.1),
        lambda x: x["choices"][0]["prompt_logprobs"][1]["11"].update(logprob=True),
    ]
    for mutate in mutations:
        fixture = copy.deepcopy(valid)
        mutate(fixture)
        try:
            validate_response(fixture)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("malformed response accepted")
    base = {"schema_version": 1, "status": "passed", "rounds": rounds,
            "prompts": descriptors(prompts), "run_nonce": "preflight", "request_settings": REQUEST,
            "probe_source_sha256": {}, "schedule": planned,
            "results": [{**copy.deepcopy(row), **p} for p in planned]}
    other = copy.deepcopy(base)
    other["run_nonce"] = "other"
    other["schedule"] = schedule(prompts, rounds, "other")
    other["results"] = [{**copy.deepcopy(row), **p} for p in other["schedule"]]
    require(all(r["mean_delta"] == 0 for r in compare(base, other)["rows"]), "paired summary error")
    for mutate in (lambda x: x["results"].pop(),
                   lambda x: x.update(status="failed"),
                   lambda x: x["prompts"][0].update(prompt_sha256="different"),
                   lambda x: x["results"][0].update(nll=99),
                   lambda x: x["results"][0].update(completion_tokens=0),
                   lambda x: x["results"][0].update(cache_salt=x["results"][1]["cache_salt"])):
        fixture = copy.deepcopy(other)
        mutate(fixture)
        try:
            compare(base, fixture)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("malformed comparison accepted")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "result.json"
        atomic_save(path, {"status": "running"})
        atomic_save(path, base)
        require(json.loads(path.read_text()) == base and len(list(Path(directory).iterdir())) == 1,
                "atomic result persistence failed")
        # Exercise the real request/persistence loop with an in-memory transport.
        # No socket or HTTP request is made, including for the injected failure.
        payloads = []
        def fake_open(request, timeout):
            payload = json.loads(request.data)
            require(all(payload[k] == v for k, v in REQUEST.items()), "request settings changed")
            payloads.append(payload)
            response = io.StringIO(json.dumps(valid))
            response.status = 200
            return response
        args = SimpleNamespace(output=Path(directory) / "success.json", rounds=2,
                               base_url="http://offline.invalid", timeout=1)
        with patch.object(urllib.request, "urlopen", fake_open), redirect_stdout(io.StringIO()):
            run(args)
        finished = json.loads(args.output.read_text())
        require(len(summarize(finished)) == 10 and len(finished["results"]) == 20,
                "mock request loop did not complete")
        require(len({p["cache_salt"] for p in payloads}) == 20, "mock requests reused cache salt")
        by_id = {p["id"]: p["text"] for p in prompts}
        require(all(payload["prompt"] == by_id[step["prompt_id"]]
                    for payload, step in zip(payloads, finished["schedule"], strict=True)),
                "actual request prompt differs from hashed prompt")
        args.output = Path(directory) / "failure.json"
        attempts = 0
        def failing_open(request, timeout):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                raise RuntimeError("synthetic transport failure")
            return fake_open(request, timeout)
        with patch.object(urllib.request, "urlopen", failing_open), redirect_stdout(io.StringIO()):
            try:
                run(args)
            except RuntimeError:
                pass
            else:
                raise AssertionError("transport failure did not propagate")
        failed = json.loads(args.output.read_text())
        require(failed["status"] == "failed" and len(failed["results"]) == 1
                and failed["failure"]["next_request_index"] == 1, "partial results were lost")
    print(json.dumps({"status": "passed", "mode": "cpu_preflight", "http_requests": 0,
                      "prompts_identical_to_original": 10, "rounds": rounds,
                      "unique_request_salts": len(planned), "malformed_fixtures_rejected": rejected,
                      "mocked_successful_requests": 20, "partial_failure_preserved": True,
                      "script_sha256": sha(Path(__file__).read_bytes()), "prompts": descriptors(prompts)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--compare", nargs=2, type=Path, metavar=("BASE", "CAND"))
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--base-url", default=os.environ.get("BENCH_URL", "http://127.0.0.1:8132"))
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    require(args.rounds > 0 and math.isfinite(args.timeout) and args.timeout > 0,
            "rounds and timeout must be positive")
    if args.preflight:
        preflight(args.rounds)
    elif args.compare:
        data = [json.loads(p.read_text()) for p in args.compare]
        result = compare(*data)
        result["input_files"] = [{"path": str(p), "sha256": sha(p.read_bytes())} for p in args.compare]
        print(json.dumps(result, indent=2, allow_nan=False))
        print("prompt | baseline mean [min,max] | candidate mean [min,max] | delta | delta / larger range", file=sys.stderr)
        for r in result["rows"]:
            x, y, ratio = r["baseline"], r["candidate"], r["delta_over_larger_range"]
            spread_text = f"{ratio:+.3f}" if ratio is not None else "undefined (zero range)"
            print(f"{r['prompt_id']} | {x['mean']:.6f} [{x['min']:.6f},{x['max']:.6f}] | "
                  f"{y['mean']:.6f} [{y['min']:.6f},{y['max']:.6f}] | {r['mean_delta']:+.6f} | "
                  f"{spread_text}", file=sys.stderr)
        print(result["interpretation"], file=sys.stderr)
    else:
        run(args)


if __name__ == "__main__":
    main()
