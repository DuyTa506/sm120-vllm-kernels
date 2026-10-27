"""Print paired serving tables only for complete, compatible experiment runs.

Usage: python3 bench/compare_serving_matrix.py BASE_RUN CANDIDATE_RUN
Inputs are qwen_experiment.sh result directories; no server/GPU access.
"""
import json
from pathlib import Path
import sys


def load(path):
    root = Path(path)
    if (root / "exit_code.txt").read_text().strip() != "0":
        raise ValueError(f"experiment did not complete successfully: {root}")
    matrix = root / "serving/serving_matrix_all"
    data = json.loads((matrix / "summary.json").read_text())
    manifest = json.loads((matrix / "manifest.json").read_text())
    if data["status"] != "passed" or len(data["results"]) != 17:
        raise ValueError(f"incomplete serving matrix: {root}")
    config = dict(line.split("=", 1) for line in (root / "config.txt").read_text().splitlines())
    return root, matrix, data, manifest, config


def counters(path):
    values = {}
    for name in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
        prefix = f"vllm:spec_decode_{name}_total"
        values[name] = sum(float(line.split()[-1]) for line in path.read_text().splitlines()
                           if line.startswith(prefix + "{") or line.startswith(prefix + " "))
    return values


def acceptance(matrix, scenario):
    pre = counters(matrix / f"{scenario}.measured.metrics_before.txt")
    post = counters(matrix / f"{scenario}.measured.metrics_after.txt")
    delta = {key: post[key] - value for key, value in pre.items()}
    if any(value < 0 for value in delta.values()):
        raise ValueError("MTP counters reset during the measured phase")
    return (delta["num_accepted_tokens"] / delta["num_draft_tokens"]
            if delta["num_draft_tokens"] else None)


def change(before, after):
    return f"{100 * (after / before - 1):+.1f}%" if before else "—"


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    for field in ("config", "gpu_util"):
        if a[4][field] != b[4][field]:
            raise ValueError(f"incompatible {field}")
    if (a[0] / "image.txt").read_text() != (b[0] / "image.txt").read_text():
        raise ValueError("different container images")
    if a[3]["scenarios"] != b[3]["scenarios"]:
        raise ValueError("different workload definitions/seeds")
    rows_b = {r["scenario"]: r for r in b[2]["results"]}
    print(f"Baseline `{a[0].name}`; candidate `{b[0].name}`; GPU utilization {a[4]['gpu_util']}.")
    print("\nPositive rate changes mean faster; negative latency changes mean faster.")
    for suite in ("throughput", "prefill", "decode"):
        print(f"\n**{suite}**\n")
        print("| task | total tok/s baseline / candidate | change | TTFT p50 ms baseline / candidate | change | ITL p50 ms baseline / candidate | output tok/s change | MTP acceptance baseline / candidate |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for x in a[2]["results"]:
            if x["suite"] != suite:
                continue
            y = rows_b[x["scenario"]]
            rates = [acceptance(run[1], x["scenario"]) for run in (a, b)]
            mtp = " / ".join(f"{100*r:.1f}%" if r is not None else "—" for r in rates)
            print(f"| {x['scenario']} | {x['total_tok_s']:.1f} / {y['total_tok_s']:.1f} | "
                  f"{change(x['total_tok_s'], y['total_tok_s'])} | "
                  f"{x['med_ttft_ms']:.1f} / {y['med_ttft_ms']:.1f} | "
                  f"{change(x['med_ttft_ms'], y['med_ttft_ms'])} | "
                  f"{x['med_itl_ms']:.2f} / {y['med_itl_ms']:.2f} | "
                  f"{change(x['out_tok_s'], y['out_tok_s'])} | {mtp} |")
    print("\nThese are single-run task comparisons. Dataset seeds fix prompts; sampling follows the request/model settings recorded with each run.")


if __name__ == "__main__":
    main()
