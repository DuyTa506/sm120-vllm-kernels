# System-level candidates beyond kernel tuning

CPU source audit, 2026-10-28. No GPU experiment or measured gain in this document. Installed sources refer to `/usr/local/lib/python3.12/dist-packages/vllm` inside `qwen-prefill-dev`; inspect actual mounted sources again before a future integration. Calibration candidate remains isolated in `kernels/moe-layer-dispatch`.

## Baseline and prior work

`scripts/serve.sh:66` fixes MTP depth2, with `NOSPEC=1` as a full disabling ablation. `CONFIG=next` uses max-num-batched-tokens16384, max-num-seqs64, chunked prefill, async scheduling, block-size2176 and the explicit `scripts/cudagraph_sizes.txt` ladder. That ladder already contains1024/1536/2048 and512-spaced shapes through8192. Changing to those existing graph shapes alone is not new work.

The prior sequential plan explicitly records that target NVFP4 graph padding is already masked; decode-route capture distinguishes verifier rows (including rejected candidates) from accepted tokens. Rejected verifier candidates are valid required computation and must never be treated as padding. Historical GDN-replay work assumes MTP2 and three state slots; it is not evidence that an arbitrary dynamic-depth policy composes safely. Existing max-batch/graph tuning, NOSPEC ablations, cache-chunk work and routing capture must not be presented as newly invented optimizations.

## 1. Prefill-pressure-aware MTP depth, then acceptance feedback

**Already supported upstream:** `config/speculative.py:491` exposes `num_speculative_tokens_per_batch_size`; `v1/core/sched/scheduler.py:307-314` builds its lookup and1485-1490 selects depth from the number of scheduled requests. `v1/spec_decode/dynamic/utils.py:77` normalizes a static schedule, clamps to configured maximum depth, and fills gaps by carrying the previous value. The standard MTP proposer inherits `target_dependent_ar/speculator.py`; `propose:191` accepts a dynamic count and decode loops at495/599 honor that count. V2 `model_runner.py:2239-2268` passes the selected depth, masks unused draft slots and hands only that depth to the handler.

**New hypothesis for this stack:** the same batch count can represent pure decode or a long-prefill mixed step; reducing depth only under prefill pressure may release GPU time for TTFT without permanently sacrificing low-concurrency decode acceptance benefits. Then add lagged acceptance feedback only if static pressure-based evidence justifies it. Request count alone does not describe verifier token volume or prefill work.

Important implementation detail: dynamic depth0 still executes the initial MTP draft-prefill forward (`target_dependent_ar/speculator.py:274-294`) before returning at296-299. It is therefore not equivalent to `NOSPEC=1`. Skipping that initial forward naively can leave draft KV/state stale when speculation resumes. Initial experiment should vary2→1 and include a separately restarted NOSPEC server only as a diagnostic upper bound, not pretend dynamic0 removes all draft cost.

**Smallest falsifiable sequence:** first no-code configuration ablation, fixed maximum2 plus schedule `[[1,8,2],[9,64,1]]`, against fixed2 on identical low/high-concurrency workloads; boundaries are test choices, not calibrated recommendations. Run long-doc c2 and code/QA at representative decode concurrency, plus one long-doc arrival while decode requests remain active. Record p50/p95 TTFT, ITL, accepted tokens per verifier step, target/draft time and total throughput. Fixed1 gives a useful control. Keep graph captures/config-cache namespaces separate. Reject if mixed-load TTFT does not improve enough to offset decode throughput loss.

If promising, the exact new scheduler hook is immediately before SchedulerOutput creation at1485: cap the next proposal depth when this scheduled batch includes prefill work (use host request `num_computed_tokens`/prompt length and `num_scheduled_tokens`, not a device read). This controls upcoming proposals; existing verifier candidates must still be verified and it does not retroactively shrink the current step. Acceptance feedback should use counters already returned asynchronously through scheduler output handling; smooth over completed steps with hysteresis, never add a GPU synchronize for a policy decision. Validate alternating depths, rejection, cancellation, prefix hits, preemption and GDN state/checkpoint behavior before stacking with replay patches. Concurrency-adaptive depth is existing functionality; prefill-pressure and acceptance-aware control is the untested addition here.

## 2. Backend-aware chunk allocation without token padding

**Observed source:** `scheduler.py:640-655` computes `long_prefill_token_threshold`. It deliberately disables that cap when only one eligible request exists; optional adaptive mode floors it at a fair share of input budget. Scheduling clamps at708-709 and1155-1156. Thus setting only the long-prefill threshold does not bound a lone long prompt. Global max-num-batched-tokens remains the reliable first ablation knob. MoE backend dispatch depends on total executing token shape, potentially graph-padded, not the length of one request's chunk.

**New hypothesis:** choose actual useful prefill tokens to land the *aggregate* mixed-step token count near an independently measured backend sweet spot, accounting for current verifier/decode tokens. Do not manufacture dummy tokens merely to cross1024. Avoid reducing every long prefill to1024 solely because it enables b12x: larger chunks may amortize overhead and improve TTFT, whereas smaller chunks can protect decode ITL but add scheduler/state-checkpoint work.

**Smallest falsifiable experiment:** wait for the per-layer backend calibration; select two real total-token budgets supported by the existing graph ladder (initial2048 vs4096, plus current16384 baseline). Replay the same long-doc c2 and mixed-arrival scenario. Log actual total/prefill/verify tokens, graph padded size, backend chosen and iteration duration, in addition to TTFT/ITL. A raw GEMM gain without request-level benefit fails. Check final tail chunks around1023/1024; do not assume the max budget predicts their execution shape.

If a useful discontinuity survives, prototype a host scheduler rounding-down of the *remaining prefill allowance* to a measured total-step bucket after reserving decode work. Preserve fairness, progress for short tails, KV allocation alignment, prefix checkpoints, max token budget and lookahead accounting; never let rounding produce zero progress or wait indefinitely for a fuller batch. The existing cache-granularity work already shows that extra scheduler splits have a cost. This candidate is scheduling tied to measured backend costs, not another cache-block-size change.

## 3. Graph-padding waste: distinguish masked target from draft

**Existing correct target behavior:** V2 `model_runner.py:1362-1367` constructs true/padded token masks; forward context receives it around2026. `fused_moe/router/fused_topk_router.py:21-58` passes that mask into routing; grouped router356-360 explicitly writes weight0/id-1. Scalar b12x handles invalid IDs; do not reimplement target masking and call it a new optimization. Also distinguish a zero route weight from skipping expert computation: a fallback may remap masked IDs to0 and require downstream behavior inspection.

**Concrete untested gap:** `target_dependent_ar/speculator.py:_prefill:377-390` initializes `is_padding=None` and supplies a padding buffer only in the PCP path. On this single-GPU configuration that means target and draft forward-context masking differ. This is source evidence for a candidate, not proof of expensive padded draft experts in the measured workload. The existing plan notes draft experts are BF16, so NVFP4 masking changes would target the wrong operator.

**Smallest falsifiable experiment:** reuse raw route capture to log true and padded row counts separately for target verification, draft-prefill and later draft steps at request counts adjacent to graph boundaries. Weight the extra expert routes by observed expert footprint; report both graph padding and necessary rejected candidates separately. If padded draft work is negligible, stop this candidate. A prior sampled prefill had zero graph padding, so do not infer broad waste from the capture ladder alone.

If material, first try a denser graph-size ladder only around observed frequent shapes, measuring capture time/memory and steady-state throughput; that changes no mathematical state. Then an isolated draft forward-context mask prototype should derive a stable per-token validity buffer from actual draft input metadata and pass it through the existing router contract. Test BF16 expert backend support for negative IDs before enabling the mask. Live rows, draft logits/samples, KV slot mappings and GDN/conv state must match baseline; padded state slots must remain untouched across replay, request-count changes, rejection and prefix resume. Do not dynamically slice tensors to actual size inside captured graphs or mask necessary verifier candidates.

## Priority and decision rule

1. Configuration-only dynamic2→1 test under controlled prefill interference, after the current GPU queue. This is the lowest integration-cost new systems experiment.
2. Useful-token chunk budgets informed by completed backend calibration; evaluate TTFT and decode ITL together.
3. Instrument draft padding before implementing masking. Target mask removal is not a valid optimization.

None of these has a measured gain in this audit. Keep all runtime changes opt-in and separate until task-level benefit survives paired runs. Agentic remains the final combined-stack experiment, not a per-candidate test.
