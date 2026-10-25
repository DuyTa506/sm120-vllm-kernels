"""Print a markdown table of session_bench result files: summarize.py results/*.json"""
import json, statistics, sys

print("| run | conc | later turns | prefill mean | prefill median | TTFT mean | TTFT p50 | TTFT p90 | spec accept | preempt | wall s |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for f in sys.argv[1:]:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    s = d.get("summary", {})
    if "prefill_mean" not in s:
        continue
    md = s.get("metrics_delta", {})
    print(f"| {f.split('/')[-1].removesuffix('.json')} | {s['conc']} | {s['later_turns']} | {s['prefill_mean']:.0f} | "
          f"{s['prefill_median']:.0f} | {s['ttft_mean_ms']:.0f} | {s['ttft_p50_ms']:.0f} | {s['ttft_p90_ms']:.0f} | "
          f"{s.get('spec_accept_rate', 0):.3f} | {md.get('vllm:num_preemptions_total', 0):.0f} | {s['wall_s']:.0f} |")
