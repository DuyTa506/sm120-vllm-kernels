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
| Humming NVFP4, RN scale rounding | M1024 hot-layer expert-call time is about 6.5% lower at L0, 10.9% at L19 and 7.5% at L39. L19/M4096 remains about 33.7% slower. | Shape-specific candidate across three checked layers; no global replacement and no measured serving gain yet. |
| Humming numerical diagnosis | Matching scale rounding makes FC1 quantized bytes identical to vLLM. L19/M1024 output error versus FP32 is 14.490%, versus CUTLASS 14.494%. Remaining backend differences persist. | Broader layer/shape and model-quality checks remain. |
| Native MoE task order | L19/M4096 hot-layer full-call time falls about 3.9%; diagnosed weighted contributions match before scatter. Live pair3 verifies 40 candidate graph nodes: target kernel sum 17.226 → 17.026 ms, but nonstream request median 69.000 → 76.754 ms. | Unselected. Two of ten common Inductor configs differ; this pair establishes no serving win or isolated causal effect. |
| DeepGEMM rowwise FP8 | Small actual QKVZ operand cases improve 2.7–10.1% over the faster existing comparator; actual fused graph execution is proven. | Short serving results are mixed, with compiler/quality/lifecycle confounds. Unselected. |
| Native swap-AB | Two matrices give identical off/on outputs in all 16 comparisons; QKVZ-shaped cases improve 2.3–4.1%. | Confirms an existing native policy, not a new model patch. |
| Tiny rowwise 64×16 tile | L0/M3: 19.62 µs, about 11.2% slower than AUTO and 16.1% slower than existing skinny. Numerical/replay checks pass. | Rejected for the measured case. |
| PDL | Focused pair pins the same 18 compiler identities/configs. Prefill TTFT falls 0.26–3.30%; decode c1/c4 rises 0.67%/4.28%. | Unselected, one pair. Later c16/c64 and NLL have a possible external-context overlap; numerical variation remains unresolved. |
| DP4A expert reuse | M1/M8 numerical screens pass input-byte, quantized-operand, graph/eager and normal/instrumented checks; M8 includes eight tokens sharing all eight experts. | No timing or model-quality claim. Concurrent execution remains unvalidated. |
| Split B12x prefill | L19/M1024 full-domain numerical and changing-route graph checks pass exploratory gates, including all-empty then restored input. | Full-call timing pending. Atomic output variation and a small phase1 quantization difference remain. |
| Workspace tile bands | Actual M2048 wrapper uses M128 despite the functional selector choosing M64; saved routing gives 46.7% versus 65.8% useful rows. | New integration experiment. Smaller tiles also increase tile count; geometry is not a speedup measurement. |
| GDN state replay | Current-stack shadow checks pass, including intermediate checkpoint boundaries. The first pressure workload produced zero preemptions. | Stronger pressure coverage is still required before timing/promotion. |

The [round 2 snapshot](../results/round2_snapshot/manifest.json) contains hash-verified
copies of completed numerical/micro reports, not a full implementation archive. Timing scope is retained in each
report; these measurements are not model throughput or agentic results.
Experimental implementation work remains isolated. A unified, independently
switchable stack has passed CPU composition checks; its GPU evaluation is pending.
Overlapping GEMMs need explicit dispatch priority;
compatible changes to different operators can be composed.

Humming RN M1024 now completes the planned L0/L19/L39 screen. Each case uses two
disjoint captured input slices, repeated outputs, graph/eager comparison and
an explicit FP32 reference with TF32 disabled. The full expert-call timing
includes packing, quantization, GEMMs and reduction; router/top-k and initial
weight preparation are outside it. The two timing passes are:

| layer | scalar µs | Humming RN µs | Humming time change |
|---|---:|---:|---:|
| 0 | 329.936 / 329.744 | 308.832 / 307.792 | −6.5% |
| 19 | 315.168 / 314.048 | 280.416 / 280.512 | −10.9% |
| 39 | 310.896 / 310.384 | 287.200 / 287.184 | −7.5% |

The percentages compare means of the two pass medians, not independent startup
trials. Numerical differences remain: for example L39/offset0 error versus FP32
is 9.294% for Humming and 9.252% for CUTLASS; the next captured slice reverses
that ordering. These are diagnostic checks, not model-quality equivalence.
Actual vLLM dispatch, preparation/memory costs and cold/multi-layer effects
remain unmeasured for Humming.

Task-order pair3 completed both source, activation, fixed-FlashInfer and cleanup
gates. Earlier class-only substitution could reuse a baseline native cached
object; pair3 qualifies native cache keys by implementation source and proves
the actual candidate GPU symbols. This finding applies to that experiment,
not automatically to previously measured stacks. Its [full pair summary](../results/round2_snapshot/moe_task_order_serving_pair3.json)
retains the compiler and numerical limitations.

The [focused PDL pair](../results/round2_snapshot/pdl_focused_pair1.json) completed
six prefill cases, four greedy decode cases and 50 cold NLL probes per control.
All prefill and c1/c4 timing finished before the reported external-context
window. During later checks, our source-inspection import in another container
unexpectedly initialized CUDA; we flagged the conservative overlapping interval
instead of treating the entire run as isolated. Compiler matching is proven,
but these modest, mixed timing changes require an independent repeat. NLL
within-control variation persists with pinned configurations, so neither
numerical equivalence nor a PDL-induced quality change is established.

The [DP4A M1 report](../results/round2_snapshot/dp4a_numerical_m1.json) uses two
captured input slices. FC1 input payload and scales match vLLM byte-for-byte;
FC1 and FC2 agree with FP32 references using the same quantized operands at
relative L2 1.54e-7 and 1.20e-7. Whole-output error against the dequantized-weight
FP32 reference is still about 18–21%, similar to current CUTLASS in these two
slices: the small per-stage errors do not remove NVFP4 quantization error.
Two workspace checks are sequential isolation checks, not concurrent-stream
coverage. There is no latency measurement yet.

The [M8 follow-up](../results/round2_snapshot/dp4a_numerical_m8.json) exercises
31 and 40 unique experts in two real slices (up to five and seven tokens sharing
an expert), plus a controlled case where all eight tokens share eight experts
with permuted route slots. Reuse on/off remains bitwise identical, as do normal
versus instrumented outputs and each path's graph/eager outputs. This establishes
cross-token reuse correctness for the checked cases, without measuring its speed.

The [split-prefill screen](../results/round2_snapshot/b12x_split_numerical_l19_m1024.json)
checks all 8,192 routed rows at L19/M1024. Packed input and SFA publication match
exactly. Phase1 scales match the reference byte-for-byte; one of 4,194,304 decoded
intermediate values differs in the original input, with relative L2 9.14e-5.
Large packed-byte mismatch counts remain recorded, chiefly consistent with
signed-zero encoding; the reference payload was not retained for a pairwise
classification. Phase2 output differs from fixed-order FP64 reduction of the
actual intermediate by about 0.38–0.39% relative L2; graph/eager differences are
about 0.18–0.19%, consistent with variable BF16 atomic accumulation order.
These pass explicitly exploratory thresholds, not production quality gates.

Next: prepare shape-specific Humming integration, complete pending
PDL/skinny/state-replay checks, and test the mechanisms identified in the
[external kernel review](EXTERNAL_KERNEL_REVIEW.md): split B12x prefill,
expert-weight reuse for decode and same-checkpoint Marlin. Then benchmark the
selected combined stack. Agentic remains the final combined-stack stage.

## Regenerate the chart

```bash
python -m pip install matplotlib
python docs/plot_progress.py
```

The script reads saved benchmark artifacts and writes light/dark SVG and PNG,
a light PDF, and a JSON containing exact values and source hashes. It does not
run a benchmark or substitute microbenchmark percentages for serving results.
