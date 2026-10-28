# Measured progress — 2026-10-28

The current measured stack is **optimized → next → next + scalar b12x**.
`next` remains the recommended default. Scalar is an opt-in prefill improvement
with the quality and mixed-workload limits recorded below. Round 2 is ongoing;
its kernel timings are not added to the serving gains in the chart.

![Measured prefill progression](img/progress-prefill-light.svg)

The historical `optimized` values come from `results/prefill/ttft_optimized.txt`.
The newer `next` and `scalar` values come directly from eight-request measured
JSON files in `next_matched_3` and `scalar_matched_3`. This is a progress reference,
not a fresh three-way controlled trial. Source hashes and exact values are in
[the chart data](img/progress-prefill-data.json).

## Completed serving work

- `next` adds patches 03–05 to `optimized`: one-kernel FP8 attention prefill,
  smaller attention pages inside the hybrid cache block, and fused GDN
  convolution preparation. The historical same-session GPU span comparison is
  136.8 → 125.1 ms at 8k and 573.9 → 551.4 ms at 32k.
- The scalar b12x composition fixes effective scales, shares MoE scratch across
  layers, and avoids unnecessary scale expansion in its supported path. Its
  fixed-FlashInfer comparison completes all 17 serving tasks. TTFT versus next
  falls 18.8% at 8k, 9.4% at 16k and 7.1% at 32k. Three common Inductor
  selections differ; these are configuration-level results.
- Seven mixed-workload throughput changes range from −1.4% to +5.2%.
  The completed final agentic comparison covers 5,772 turns and 987,945 output
  tokens per configuration, with modest, mixed effects. It does not establish
  the same benefit as the cold-prefill screen.
- Repeated NLL probes retain numerical variability and quality flags. Neither
  correctness at a kernel boundary nor the limited task probes establish
  general model-quality equivalence. Scalar remains opt-in.

References: [next](../results/next/SUMMARY.md),
[scalar serving](../results/qwen_sequential/scalar_vs_next_matched_3.serving.md),
[scalar dispatch](../kernels/moe-nvfp4/SCALAR_DISPATCH.md),
[final agentic](../results/qwen_sequential/final_agentic_analysis.md).

## Round 2: confirmed observations, not selected defaults

| candidate | measured observation | disposition |
|---|---|---|
| Humming NVFP4, RN scale rounding | L19/M1024: 280.4–280.5 µs versus scalar 314.0–315.2 µs, about 10.9% lower hot-layer expert-call time. At M4096: 580.0 µs versus 433.7–433.9 µs, about 33.7% slower. | Shape-specific candidate; no global replacement and no measured serving gain yet. |
| Humming numerical diagnosis | Matching scale rounding makes FC1 quantized bytes identical to vLLM. L19/M1024 output error versus FP32 is 14.490%, versus CUTLASS 14.494%. Remaining backend differences persist. | Broader layer/shape and model-quality checks remain. |
| Native MoE task order | L19/M4096 hot-layer full-call time falls about 3.9%. All 59,198 weighted contributions across 2,048 output columns match bit-for-bit before atomic scatter in the diagnosed case. | Live vLLM graph-prefill comparison is in progress; no selected stack gain yet. |
| DeepGEMM rowwise FP8 | Small actual QKVZ operand cases improve 2.7–10.1% over the faster existing comparator; actual fused graph execution is proven. | Short serving results are mixed, with compiler/quality/lifecycle confounds. Unselected. |
| Native swap-AB | Two matrices give identical off/on outputs in all 16 comparisons; QKVZ-shaped cases improve 2.3–4.1%. | Confirms an existing native policy, not a new model patch. |
| Tiny rowwise 64×16 tile | L0/M3: 19.62 µs, about 11.2% slower than AUTO and 16.1% slower than existing skinny. Numerical/replay checks pass. | Rejected for the measured case. |
| PDL | Current paired compiler/artifact controls are verified; repeated focused timing and NLL checks are queued. | Historical synthetic micro numbers are not an actual vLLM benefit. |
| GDN state replay | Current-stack shadow checks pass, including intermediate checkpoint boundaries. The first pressure workload produced zero preemptions. | Stronger pressure coverage is still required before timing/promotion. |

The [round 2 snapshot](../results/round2_snapshot/manifest.json) contains hash-verified
copies of completed numerical/micro reports, not a full implementation archive. Timing scope is retained in each
report; these measurements are not model throughput or agentic results.
Experimental implementation work remains isolated. A unified, independently
switchable stack has passed CPU composition checks; its GPU evaluation is pending.
Overlapping GEMMs need explicit dispatch priority;
compatible changes to different operators can be composed.

Next: finish task-order serving comparisons, check Humming RN on other layers at
M1024, complete pending PDL/skinny/state-replay checks, then benchmark the selected
combined stack. Agentic remains the final combined-stack stage.

## Regenerate the chart

```bash
python -m pip install matplotlib
python docs/plot_progress.py
```

The script reads saved benchmark artifacts and writes light/dark SVG and PNG,
a light PDF, and a JSON containing exact values and source hashes. It does not
run a benchmark or substitute microbenchmark percentages for serving results.
