"""Rebuild this pair's CPU-only serving and quality analysis from saved files.

Run from any directory: python3 /path/to/compare_shared_vs_next_tasks_1.py
No torch imports, server requests, or GPU access.
"""
from pathlib import Path
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from compare_serving_matrix import counters, load

BASE, CAND = "next_tasks_1", "b12x_tasks_1"
STEM = "shared_vs_next_tasks_1"
inputs = {}


def track(path):
    inputs[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read(path):
    return json.loads(track(path).read_text())


def percent(a, b):
    return 100 * (b / a - 1) if a else None


def nll(row):
    assert row["tokens"] == len(row["lps"]) + 1 and row["lps"]
    assert all(math.isfinite(x) for x in row["lps"])
    return -statistics.mean(row["lps"])


def lp_diff(a, b):
    assert (a["i"], a["tokens"]) == (b["i"], b["tokens"])
    delta = [abs(x - y) for x, y in zip(a["lps"], b["lps"], strict=True)]
    return {"mean_abs_logprob_delta": statistics.mean(delta),
            "max_abs_logprob_delta": max(delta),
            "exact_logprobs": sum(x == 0 for x in delta),
            "greedy_continuation_identical": a["gen"] == b["gen"]}


# Run the repository's compatibility/completeness checks and preserve its table.
serving = subprocess.check_output(
    [sys.executable, str(ROOT / "bench/compare_serving_matrix.py"),
     str(HERE / BASE), str(HERE / CAND)], text=True)
(HERE / f"{STEM}.serving.md").write_text(serving)
runs = [load(HERE / name) for name in (BASE, CAND)]
for root, matrix, summary, manifest, _ in runs:
    for name in ("config.txt", "exit_code.txt", "image.txt", "commit.txt",
                 "uncommitted.diff", "startup.log"):
        track(root / name)
    for name in ("summary.json", "manifest.json", "expected_output_lengths.json"):
        track(matrix / name)
    assert len(manifest["commands"]) == 34
    assert len({c["cache_salt"] for c in manifest["commands"]}) == 34
    for command in manifest["commands"]:
        argv = command["command"]
        assert "--temperature" not in argv
        assert json.loads(argv[argv.index("--extra-body") + 1])["cache_salt"] == command["cache_salt"]
        for key in ("--num-warmups", "--ready-check-timeout-sec"):
            assert argv[argv.index(key) + 1] == "0"
    assert "{'temperature': 1.0, 'top_k': 20, 'top_p': 0.95}" in (root / "startup.log").read_text()
assert runs[0][3]["run_nonce"] != runs[1][3]["run_nonce"]

rows = []
for a, b in zip(runs[0][2]["results"], runs[1][2]["results"], strict=True):
    assert a["scenario"] == b["scenario"]
    mtp = []
    for run in runs:
        before = counters(track(run[1] / f"{a['scenario']}.measured.metrics_before.txt"))
        after = counters(track(run[1] / f"{a['scenario']}.measured.metrics_after.txt"))
        delta = {key: after[key] - value for key, value in before.items()}
        assert all(x >= 0 for x in delta.values())
        delta["acceptance_rate"] = (delta["num_accepted_tokens"] / delta["num_draft_tokens"]
                                     if delta["num_draft_tokens"] else None)
        mtp.append(delta)
    rows.append({"scenario": a["scenario"], "suite": a["suite"], "baseline": a,
                 "candidate": b, "change_percent": {
                     key: percent(a[key], b[key]) for key in
                     ("total_tok_s", "out_tok_s", "med_ttft_ms", "med_itl_ms")},
                 "mtp_baseline": mtp[0], "mtp_candidate": mtp[1]})

lp = [[read(HERE / name / file) for file in ("logprobs.json", "logprobs_repeat.json")]
      for name in (BASE, CAND)]
assert all(len(dump) == 4 for pair in lp for dump in pair)
logprobs = []
for i in range(4):
    a, b = [[dump[i] for dump in pair] for pair in lp]
    an, bn = [nll(x) for x in a], [nll(x) for x in b]
    am, bm = statistics.mean(an), statistics.mean(bn)
    logprobs.append({"prompt": i, "tokens": a[0]["tokens"],
                     "baseline_nll": an, "candidate_nll": bn,
                     "baseline_mean_nll": am, "candidate_mean_nll": bm,
                     "mean_nll_delta": bm - am, "mean_nll_change_percent": percent(am, bm),
                     "baseline_repeat": lp_diff(*a), "candidate_repeat": lp_diff(*b),
                     "cross_run_pairs": [lp_diff(x, y) for x in a for y in b]})

long = [read(HERE / name / "long_eval.json") for name in (BASE, CAND)]
assert all(len(x["nll"]) == 6 and len(x["needle"]) == 20 for x in long)
long_nll = []
for a, b in zip(long[0]["nll"], long[1]["nll"], strict=True):
    assert (a["doc"], a["tokens"]) == (b["doc"], b["tokens"])
    long_nll.append({"doc": a["doc"], "tokens": a["tokens"], "baseline": a,
                     "candidate": b, "nll_delta": b["nll"] - a["nll"],
                     "nll_change_percent": percent(a["nll"], b["nll"])})
for a, b in zip(long[0]["needle"], long[1]["needle"], strict=True):
    assert all(a[key] == b[key] for key in ("chars", "depth", "prompt_tokens"))
needles = [{"correct": sum(x["ok"] for x in run["needle"]), "total": len(run["needle"])}
           for run in long]
checkpoint = []
for name in (BASE, CAND):
    text = track(HERE / name / "checkpoint.log").read_text()
    m = re.search(r"hits/prompt=(\d+)\s+mean TV\(top20\)=([\d.]+)\s+max TV=([\d.]+)\s+top1 agree=(\d+)/(\d+)", text)
    assert m
    checkpoint.append(dict(zip(("hits_per_prompt", "mean_top20_tv", "max_top20_tv", "top1_agree", "prompts"),
                               [float(x) for x in m.groups()])))
for name in ("bench/compare_serving_matrix.py", "bench/serving_matrix.py", "bench/qwen_experiment.sh",
             "kernels/moe-nvfp4/verify_logprobs.py", "eval/long_eval.py", "eval/ckpt_dist.py"):
    track(ROOT / name)
track(Path(__file__).resolve())
report = {"baseline": BASE, "candidate": CAND, "execution_order": [CAND, BASE],
          "config": "next", "gpu_util": "0.78", "candidate_overlay": "shared (before exact-alpha correction)",
          "serving_sampling": {"temperature": 1.0, "top_k": 20, "top_p": 0.95,
                               "request_seed": None, "dataset_seeds_match": True},
          "quality_sampling": {"temperature": 0}, "serving": rows,
          "logprob_repeats": logprobs, "long_nll": long_nll,
          "needles_baseline_candidate": needles, "checkpoint_baseline_candidate": checkpoint,
          "source_sha256": inputs}
(HERE / f"{STEM}.analysis.json").write_text(json.dumps(report, indent=2) + "\n")

lines = ["# Shared b12x versus next: completed individual tasks", "",
         "Reproduce locally (CPU only): `python3 results/qwen_sequential/compare_shared_vs_next_tasks_1.py`.", "",
         f"Baseline `{BASE}`; candidate `{CAND}` (`shared/`, before the exact-alpha correction). "
         "Candidate ran first, then baseline. Both exited 0, using the same image, CONFIG=next, GPU utilization 0.78, "
         "and 17 matching workloads/seeds. Per-phase cache salts separate warmups, scenarios, and runs; "
         "requests within shared-prefix phases retain intentional prefix reuse. Exact output lengths were validated by the harness.", "",
         "The [serving table](shared_vs_next_tasks_1.serving.md) contains all task rates, TTFT, ITL, and measured-phase MTP acceptance. "
         "The [JSON analysis](shared_vs_next_tasks_1.analysis.json) contains raw values, differences, MTP counter deltas, and source SHA256s.", "",
         "## Serving and MTP limits", "",
         "The seven throughput scenarios improve 0.2–5.2% in this pair. Single-request prefill median TTFT improves "
         "0.8–14.3% (8k: 191.6→164.3 ms; 16k: 384.5→329.5 ms; 32k: 866.3→756.6 ms). "
         "Decode output throughput changes +10.3%, −3.0%, +2.2%, −1.7% at c=1/4/16/64. "
         "These are one run per variant, not confidence intervals or an adoption decision.", "",
         "Serving commands omit temperature and request sampling seed. Both startup logs report model defaults "
         "temperature=1.0, top_k=20, top_p=0.95. Dataset seeds fix prompts, not generated token paths. "
         "For decode c1, MTP acceptance rises 44.1→53.5% while median ITL is 6.52→6.56 ms; "
         "c4 acceptance falls 64.1→61.1% with ITL 8.21→8.32 ms. Thus these rates do not isolate kernel execution. "
         "The earlier short-pair ≈6% decode loss does not reproduce uniformly; explicit-greedy repeated A/B is still needed. "
         "No throughput change is attributed to the load-time 1-ULP alpha differences.", "",
         "## Quality: unresolved regressions", "",
         "Quality probes explicitly request temperature=0. Prompt-logprob requests skip prefix-cache reads in the installed vLLM, "
         "so the two repeated dumps test cold prefill. They are not deterministic even within one server. "
         "Neither server has an identical greedy continuation on any of its four repeated prompts (0/4 each). "
         "That variability limits attribution; it does not establish quality equivalence.", "",
         "| prompt tokens | next NLL, two cold calls | shared NLL, two cold calls | change in two-call mean |", "|---:|---:|---:|---:|"]
for row in logprobs:
    lines.append(f"| {row['tokens']} | {' / '.join(f'{x:.5f}' for x in row['baseline_nll'])} | "
                 f"{' / '.join(f'{x:.5f}' for x in row['candidate_nll'])} | {row['mean_nll_change_percent']:+.1f}% |")
lines += ["", "The 3,443-token mean NLL worsens by 0.05497 (+15.4%); within-run repeat shifts are −0.04996 "
          "for next and +0.06751 for shared. At 7,604 tokens the mean increases by 0.00491 (+0.9%). "
          "Mean absolute logprob differences between variants (0.121–0.209) overlap the within-run range "
          "(0.107–0.216), including the sub-1024 CUTLASS-only prompts. Two repeats cannot rule out regression.", "",
          "Long-document NLL has only one sample per document/length:", "",
          "| document | scored tokens | next NLL | shared NLL | change |", "|---|---:|---:|---:|---:|"]
for row in long_nll:
    lines.append(f"| {row['doc']} | {row['tokens']} | {row['baseline']['nll']:.5f} | {row['candidate']['nll']:.5f} | {row['nll_change_percent']:+.1f}% |")
lines += ["", "Flag the 6,349-token licenses regression: 0.10042→0.10627 NLL (+5.8%), with last-half NLL "
          "0.13368→0.14683 (+9.8%). Both needle probes are 20/20. Checkpoint hot-versus-cold mean TV over "
          "top-20 tokens worsens 0.1493→0.1710, maximum 0.4203→0.4629, and top-1 agreement falls 11/12→9/12 "
          "at the same 6,709 prefix hits per prompt. This is within-server hot/cold consistency, not direct "
          "candidate-versus-baseline TV or full-vocabulary TV. It also needs repetition.", "",
          "The candidate remains optional. Exact-alpha and scalar experiments have separate artifacts; these serving/quality "
          "numbers do not validate either later variant. Agentic evaluation remains reserved for the combined configuration.", ""]
(HERE / f"{STEM}.analysis.md").write_text("\n".join(lines))
print(json.dumps({"status": "passed", "serving_tasks": len(rows), "logprob_prompts": len(logprobs),
                  "long_nll_cases": len(long_nll), "needle_cases_per_run": 20,
                  "quality_status": "unresolved_regressions", "source_files_hashed": len(inputs)}))
