# Measured progress — 2026-10-28

The current measured stack is **optimized → next → next + scalar b12x**.
`next` remains the recommended default. Scalar is an opt-in prefill improvement
with the quality and mixed-workload limits recorded below. Round 2 is ongoing;
its kernel timings are not added to the serving gains in the chart.
The current handoff is [HANDOFF.md](HANDOFF.md). The first NCU attempt failed
with `ERR_NVGPUCTRPERM`; the user subsequently collected attempt2 successfully.
The [verified counter findings](PREFILL_COUNTER_FINDINGS.md) show 311.68 µs and
1.403 TB/s for the native scalar L19/M1024 kernel. No new serving patch is promoted.
The [CUDA profiling workflow](CUDA_PROFILING_WORKFLOW.md) documents the next
diagnostic process; its addition introduces no new benchmark result.

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
| DP4A expert reuse | Numerical screens pass checked input/operand and replay contracts. At real M3, preplanned calibrated FI direct is about 24.6 µs, DP4A reuse-on about 32–35 µs, and current scalar/CUTLASS about 49 µs. | Unselected against the stronger FI comparator. Historical M1 comparisons used a weaker FI call path; no serving or model-quality claim. |
| Prepared FI direct | L19/M3 complete-call time is 24.63 µs versus current CUTLASS 49.26 µs. Live vLLM MTP-2 tracing proves native direct and sanitizer execution in M3. | Serving pair remains unselected: raw c1 throughput +9.6%, c4 −3.5%, with four of thirteen active Inductor selections different. |
| Split B12x prefill | Actual L19 full-call timing is about 17.6% slower at M1024 and 47.4% slower at M4096 than current scalar. Pre/post arithmetic and route-domain checks pass exploratory gates. | Initial split implementation unselected at both shapes. Component measurements are diagnostic and cannot be summed into a full-call claim. |
| Workspace tile bands | L19 full-call screen: M1024/tile32 is about 1.9% slower; M2048/tile64 about 27.1% slower; unchanged M4096/tile128 is within 0.32%. | Unselected; scratch rises from 607.8 MB to 1,400.9 MB. Native M128 already skips inactive M16 MMA fragments, so padding ratios are not compute-saving estimates. |
| Inactive operand-load guards | Keeping M128 but adding warp-uniform guards makes L19/M1024 about 2.8× slower and M4096 about 3.4× slower. Compiled stack grows 80 → 688 bytes, with many additional local load/store instructions. | Rejected in this form. Source-level work reduction does not ensure better generated code. |
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
Actual vLLM dispatch, preparation costs and cold/multi-layer effects remain
unmeasured for Humming. The [CPU integration estimate](../results/round2_snapshot/humming_serving_cpu_preparation.json)
adds **16.875 GiB** for duplicate packed weights and local scales across 40 layers,
before scratch and conversion peak. Actual allocation and KV-capacity impact are
unmeasured. Standard serving tasks have not yet been shown to reach the exact
M1024 eligibility bucket; a dedicated M1024 request would prove dispatch, not a
representative serving benefit.

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
coverage. These numerical reports themselves contain no latency measurement.

The [M8 follow-up](../results/round2_snapshot/dp4a_numerical_m8.json) exercises
31 and 40 unique experts in two real slices (up to five and seven tokens sharing
an expert), plus a controlled case where all eight tokens share eight experts
with permuted route slots. Reuse on/off remains bitwise identical, as do normal
versus instrumented outputs and each path's graph/eager outputs. This establishes
cross-token reuse correctness for the checked cases, without measuring its speed.

The [historical M1 timing screen](../results/round2_snapshot/dp4a_micro_m1_historical.json)
is not evidence that DP4A beats the best available FI route: its calibrated FI
call still converted fixed scales during each invocation. The subsequent
[M3 batched screen](../results/round2_snapshot/dp4a_micro_m3_batched.json) includes
the preplanned direct path with the correct multiplier-scale ABI. On two real
slices, that path takes 24.59–24.63 µs, DP4A reuse-on 31.86–34.88 µs, and current
scalar/CUTLASS 49.19–49.26 µs. Thus beating the current CUTLASS path does not
establish a win over the strongest comparator. These are repeated-input layer
graph timings, excluding original input/routing copies and weight preparation;
no vLLM speedup or concurrent execution claim follows.

The later M8 screen exposed an invalid FI-auto/static comparator: it returned
zero against a nonzero FP32 reference. Agreement between two adapters did not
detect that error. Its timings are excluded; the matrix stopped for an ABI
investigation and an independent-reference sanity gate before resuming. This
does not establish a fault in the native static kernel.

The [static scale verification](../results/round2_snapshot/fi_static_scale_l19_m8.json)
confirms the existing contract documented in [the original small-M study](../kernels/moe-nvfp4/README.md#4c-scale-semantics-in-flashinfer-b12x-flashinfer-issue):
the MMA path needs calibrated divisor scales, whereas direct uses multipliers.
Our newer comparator incorrectly used the same inversion flag for both paths.
Corrected M8 static input payload and scale bytes match vLLM exactly on both
captured slices and a restored-input replay. Output error versus FP32 is
0.1342/0.1424, versus CUTLASS 0.1345/0.1437. This corrects the comparator;
it is not a new native-kernel fix or a timing result.

The [safe-direct checks](../results/round2_snapshot/fi_direct_padding_checks.json)
cover 56 fixtures across M1/M3/M4/M8, including invalid routes, NaNs in skipped
tokens, duplicate experts, changed inputs and graph replay. The adapter maps
padding routes to zero-weight valid IDs and clears fully skipped input rows;
the native direct kernel itself assumes valid expert IDs. Eager and graph CUDA
traces contain both the sanitizer and native direct kernel. A subsequent
[live-pointer check](../results/round2_snapshot/fi_direct_live_pointer_checks.json)
passes 12 transitions across distinct input/output allocations at M1/M4,
including interleaved graphs and eager calls. These checks are sequential;
they do not establish concurrent-stream safety or model-quality equivalence.

The [M4 complete-call screen](../results/round2_snapshot/fi_direct_m4_live_timing.json)
times that explicit native adapter with sanitizer and current runtime arguments:
26.64–26.65 µs on two real routing slices, versus scalar/CUTLASS 59.85–60.02 µs
and DP4A reuse-on 38.94–39.29 µs. Each of two reversed backend orders uses eight
samples of 50 graph replays. Original routing, weight preparation and model
execution are outside the measurement. The older invalid FI-auto comparator is
explicitly excluded; corrected micro/static comparisons follow below. The
nominal capture-size list includes M4, but does not determine the effective
MTP decode shape. The later live trace described below finds M3. This
microbenchmark result is not included in the serving chart.

The subsequent [corrected bucket comparison](../results/round2_snapshot/fi_mma_bucket_summary.json)
uses the same source/runtime/checkpoint identities at M4 and M8. Every provider
passes five numerical fixtures before timing. Both FI paths include sanitation;
the M4 micro path also includes route compaction. Two reversed orders give:

| tokens / live rows | corrected FI MMA µs | sanitized FI direct µs | current CUTLASS µs |
|---|---:|---:|---:|
| M4 / 4 | 28.70 / 28.73 | 26.67 / 26.68 | 61.22 / 61.14 |
| M4 / 3 | 32.84 / 32.84 | 26.71 / 26.67 | 49.95 / 49.99 |
| M8 / 8 | 34.80 / 34.83 | 43.02 / 43.03 | 63.49 / 63.46 |
| M8 / 7 | 34.80 / 34.79 | 43.07 / 43.06 | 61.33 / 61.32 |

Direct is the best checked M4 path, while static wins at M8. This supports
shape-specific dispatch, not replacing every small batch with direct. M4 micro
graph/eager/repeat variation reaches 0.92% relative L2; its exploratory gate is
1%, so it is not bitwise deterministic in this screen. M8 providers repeat
exactly. Padding and all-invalid outputs are exactly zero. No production quality
acceptance or end-to-end gain follows from these hot-layer checks.

The first [complete serving diagnostic](../results/round2_snapshot/fi_direct_serving_on2_applicability.json)
loads the actual model, validates 80 prepared handles across 40 layers and
completes decode c1/c4 plus prefill 2k. However, its subsequent live c1 trace
contains 108 **M3** graph replays and no candidate native-direct or sanitizer
kernel. V2 rounds the nominal capture sizes to the MTP query length of three;
the initial M1/M4 policy therefore misses this decode path. The applicability
gate fails, sources/FI controls remain unchanged, and cleanup succeeds.
These task timings are not evidence of candidate performance. The subsequent
[M3 screen](../results/round2_snapshot/fi_direct_m3_live_timing.json) passes
[18 distinct-pointer checks](../results/round2_snapshot/fi_direct_m3_live_pointer_checks.json)
across M1/M3/M4 and measures sanitized native direct at 24.63 µs versus current
CUTLASS 49.26 µs on two real routing slices, in both orders. The integration
policy now covers M1/M3/M4 and requires actual M3 candidate execution in its
serving trace before accepting applicability.

The corrected [ON3/OFF1 serving pair](../results/round2_snapshot/fi_direct_serving_pair1.json)
proves that integration: ON3's live c1 trace contains 4,960 native direct and
4,960 sanitizer graph kernels correlated to M3; OFF1 contains neither. All
three workload completion guards pass. Both runs retain fixed MTP-2, the same
132 FlashInfer tactics, and identical kernel/adapter/probe/generated sources.
The focused settings use max model length 8192, max sequences 8 and the same
nominal graph list in both modes; they are not the full production matrix.

| focused task / metric | OFF1 | ON3 | raw ON change |
|---|---:|---:|---:|
| decode c1 output tokens/s | 286.11 | 313.56 | +9.59% |
| decode c1 mean TPOT, ms | 2.333 | 2.177 | −6.69% |
| decode c4 output tokens/s | 778.44 | 751.37 | −3.48% |
| decode c4 mean TPOT, ms | 3.685 | 4.134 | +12.19% |
| prefill 2048 mean TTFT, ms | 110.26 | 109.54 | −0.66% |

These are **confounded observations, not a validated serving improvement**.
Four of thirteen active Inductor selections differ (two backbone and two draft);
the comparison filters actual loaded AOT directories, not the cumulative cache.
C4 speculative acceptance length also falls 4.36%. The single short pair uses
random-token requests, warmups and enabled prefix caching; actual cache hits
are not independently quantified. It is not a cold-prefill or quality test.

ON3 keeps its original terminal code 5 because the legacy shutdown audit
checkpoint is absent. An explicit worker RPC after all workloads and tracing
verifies the exact tactic table, worker/model identity, stable resources and
runtime sources. Cleanup succeeds. OFF1 uses the same measurement code with
a strict verifier for that explicit boundary and exits 0; the cleanup-only
harness difference is recorded, and the missing legacy checkpoint remains
visible. Actual graph integration is established, but promotion still needs
a repeat with matching active compiler selections and quality checks.

The [split-prefill screen](../results/round2_snapshot/b12x_split_numerical_l19_m1024.json)
checks all 8,192 routed rows at L19/M1024. Packed input and SFA publication match
exactly. Phase1 scales match the reference byte-for-byte; one of 4,194,304 decoded
intermediate values differs in the original input, with relative L2 9.14e-5.
A later in-run comparison directly classifies 372,496 packed-byte differences
as signed-zero-only with identical positive scales; exactly one byte remains
different after zero-sign canonicalization. The large raw mismatch count is
therefore not the count of numerical errors. Phase2 output differs from
fixed-order FP64 reduction of the actual intermediate by about 0.38–0.39% relative L2; graph/eager differences are
about 0.18–0.19%, consistent with variable BF16 atomic accumulation order.
These pass explicitly exploratory thresholds, not production quality gates.

The [complete split timing summary](../results/round2_snapshot/b12x_split_initial_summary.json)
records two reversed orders and a repeat at M1024. Scalar versus split medians
are approximately 314 versus 369 µs at M1024 and 434 versus 640 µs at M4096.
Live scalar workspaces use M128 in both component follow-ups. Standalone quant,
publisher/clears, phase1 and phase2 measurements use different cache/preparation
conditions from a contiguous call; their sums are not measured full-call time.
This split implementation is not selected for serving.

The [workspace-band screen](../results/round2_snapshot/workspace_bands_summary.json)
passes six graph transitions and its exploratory diagnostics but regresses the
changed tile sizes. The [installed-source analysis](../results/round2_snapshot/workspace_band_source_analysis.md)
shows that M128 tails already predicate inactive M16 MMA fragments in FC1 and
FC2. Lower padded-row counts do not imply the corresponding removal of tensor
core work. A narrower operand-load predicate was then tested while preserving
pipeline/barrier participation. Its [first OFF/ON pair](../results/round2_snapshot/inactive_load_pair.json)
is strongly negative: approximately 314 → 881 µs at M1024 and 437 → 1483 µs
at M4096. Eight paired output cases pass the exploratory relative-L2 gate;
runtime source hashes and captured operands match, and actual graph traces
identify the correct class in each mode. The candidate remains rejected.

The [compiled artifact analysis](../results/round2_snapshot/inactive_load_sass_summary.json)
finds unchanged 168-register allocation but stack growth from 80 to 688 bytes.
Static local-load/store instruction sites rise from 4/5 to 1,484/1,099. These are
instruction sites, not dynamic execution counts. This supports compiler
spilling/stack traffic as a likely mechanism; no stall profile establishes the
sole cause. [Artifact hashes](../results/round2_snapshot/inactive_load_artifact_digest.json)
and extracted resource reports preserve the evidence. Any repair must address
generated code before another performance claim.

The [narrow FC2 repair](../results/round2_snapshot/defined_fc2_pair.json) defines
operand registers before its six guarded copies. Its matched compiled artifacts
restore the baseline stack and local load/store counts (80 bytes, 4/5 sites).
Nevertheless, full-call timing has no positive signal: approximately 314 µs
versus 315 µs at M1024, and 437 µs versus 443 µs at M4096. All eight exploratory
paired cases pass, and runtime/input hashes match. Fixing the spill regression
does not make this load-skipping policy beneficial; the repair also remains
unselected. [Artifact identities](../results/round2_snapshot/defined_fc2_artifact_digest.json)
retain the corresponding source, output and SASS hashes.

Next: repeat the integrated small-M path with matching active compiler selections,
and collect prefill hardware counters before choosing further mechanisms from the
[external kernel review](EXTERNAL_KERNEL_REVIEW.md). The initial split and
workspace-band policies remain unselected. The current default and chart are
unchanged. Benchmark the selected combined stack only after these gates;
agentic remains the final combined-stack stage.

## Regenerate the chart

```bash
python -m pip install matplotlib
python docs/plot_progress.py
```

The script reads saved benchmark artifacts and writes light/dark SVG and PNG,
a light PDF, and a JSON containing exact values and source hashes. It does not
run a benchmark or substitute microbenchmark percentages for serving results.
