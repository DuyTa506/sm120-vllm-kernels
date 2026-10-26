"""Verifier: table of session_bench result files (adds mean prompt / completion to expose unequal workloads)."""
import json, statistics as st, sys
print("| file | turns | prompt | compl | prefill mean | prefill med | TTFT mean | p50 | p90 | accept | preempt | wall |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for f in sys.argv[1:]:
    d = json.load(open(f)); s = d["summary"]; T = [t for t in d["turns"] if t["turn"] > 0]
    print(f"| {f} | {len(T)} | {st.mean(t['prompt'] for t in T):.0f} | {st.mean(t['completion'] for t in d['turns']):.0f} | "
          f"{s['prefill_mean']:.0f} | {s['prefill_median']:.0f} | {s['ttft_mean_ms']:.0f} | {s['ttft_p50_ms']:.0f} | "
          f"{s['ttft_p90_ms']:.0f} | {s.get('spec_accept_rate', 0):.3f} | {s['metrics_delta']['vllm:num_preemptions_total']:.0f} | {s['wall_s']:.0f} |")
