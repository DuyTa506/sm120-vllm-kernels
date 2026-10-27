# CUTLASS fallback alpha arithmetic audit

The isolated `shared_exact/` candidate corrects a real representation difference;
**GPU validation passed on layers 0 and 19**, followed by all 17 serving tasks,
explicit-greedy decode and five-round cold NLL diagnostics. It does not establish the cause of the
A3/B2 short-decode throughput difference. These alpha operations run only at
weight loading; the old formula adds no per-token division. No running `shared/`, `scalar/`, serving
harness, or installed container sources were modified.

## Exact preparation chain

Installed compressed-tensors loader (`compressed_tensors_moe_w4a4_nvfp4.py`,
lines 229-246) computes FP32 weight scale2 = `1 / checkpoint_weight_global_scale`
and activation scale = `1 / checkpoint_input_global_scale`. Both backends use
`prepare_nvfp4_moe_layer_for_fi_or_cutlass`: reorder gate/up, swizzle weight block
scales, reduce activation scales with `max()` over all experts and both gate/up
shards, and repeat to `[256]`. Those processed activation tensors are retained as
`layer.w13_input_scale` / `layer.w2_input_scale`.

The quant config (`oracle/nvfp4.py:583-591`) aliases the weight scale2 parameters
as `g1_alphas/g2_alphas` and separately stores reciprocals of the processed
activation scales as `a1_gscale/a2_gscale`. Actual default
`FlashInferExperts.process_weights_after_loading` multiplies the aliased weight
parameters **in place by the original processed activation scales**.

The shared hybrid intentionally keeps raw weight scale2 for b12x. Its CUTLASS
copies previously used `s2 / a_gscale`, which introduces different rounding
from `s2 * original_activation_scale`. Therefore shared `g1_alphas/g2_alphas`
are raw s2 and cannot directly replace the fallback alphas.

The separate candidate changes only these two expressions:

```python
self._cutlass_g1 = (w13_s2 * layer.w13_input_scale).contiguous()
self._cutlass_g2 = (w2_s2 * layer.w2_input_scale).contiguous()
```

This creates independent fallback products, preserving raw s2 aliases and every
existing b12x scale/alpha assignment. Selecting this overlay is explicit; it
uses the existing FIX_SCALES/SHARED_WORKSPACE flags, with no new global patch.

## CPU checkpoint evidence

`audit_cutlass_scales_cpu.py` reads only scalar checkpoint tensors using CPU
safetensors mapping. It executes the installed default weight-processing method
extracted from its AST and the actual shared/candidate alpha assignment ASTs.
Checkpoint snapshot: `1c3f884bc99aac2524f6d49bcbac8c88401afd66`.

| Alpha | Values | Shared unequal | Shared max ULP | Exact unequal |
| --- | ---: | ---: | ---: | ---: |
| g1 | 10,240 | 2,335 (22.80%) | 1 | 0 |
| g2 | 10,240 | 2,432 (23.75%) | 1 | 0 |

All 40 layers were checked; gate/up weight and input global scales are equal in
this checkpoint. Maximum shared relative difference is `1.1898e-7`.
`results/cutlass_scale_rounding_cpu.json` records every layer and source hashes.
CUDA remained uninitialized during this audit. CPU arithmetic establishes
non-equivalence of the old formulas; the later GPU results below validate the
loaded tensors and bound output differences.

## What matches, and what remains uncertain

Input quantization is the same `ops.scaled_fp4_quant(...,
is_sf_swizzled_layout=True)`: the default calls it through no-DP/EP preparation,
while the hybrid defers it to `_apply_cutlass`. Both pass packed uint8 input,
int32 scale views, long weight views, the same activation scale tensors, and
TP/EP=1 to the same FlashInfer CUTLASS API. MMA scale conversion only creates
strided views and does not rewrite the weight block scales.

Both use the API's same defaults: `tune_max_num_tokens=8192`, no forced profile
IDs, device-selected PDL, fused finalize, and no min-latency mode. Qwen uses
unclamped SiLU; omitted optional bias/clamp/quantization arguments have the same
values as the default expert. Tactic cache inputs omit alpha values, but
independent startup tuning can still select different tactics. Actual profile
IDs in the saved full-run caches have since been compared: the
[CPU audit](../../results/qwen_sequential/prefill_runtime_cpu_audit.md) found
43/132 FlashInfer choices and 4/13 selected Inductor configs differ between
shared and exact. Per-op trace timing for those runs remains unavailable.

Both saved servers selected `FlashInfer CUTLASS Unquantized MoE` for MTP through
no-DP/EP preparation. The hybrid's unquantized oracle patch maps its requested
b12x backend to `auto`; the baseline already uses `auto`. Both log the same
FlashInfer draft-attention metadata fallback. At c1/c4, 128-token prompts mean
at most 128/512 initial prompt rows and 3/12 logical target-verification rows
with two MTP draft tokens, below the 1024-token hybrid threshold.

The old runs **did not specify temperature**. Installed vLLM 0.32 serving bench
explicitly inherits the server/model sampling default in that case; prompt
`--seed` alone does not ensure identical generations. A3→B2 acceptance length
changed 2.07497→1.94857 at c1, and 1.86254→1.92239 at c4; median ITL changed
-0.81%/-0.26% despite throughput -6.35%/-6.36%. These mixed observations do not
identify alpha rounding as the throughput cause. Sampling, MTP acceptance,
autotuning, scheduling, and host variation remain confounds. The completed full
matrices retain the original repository protocol; explicit-greedy diagnostics
are recorded separately below.

## Validation commands and later diagnostic

CPU checks (no GPU/HTTP requests):

```bash
sg docker -c 'docker exec -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 qwen-prefill-dev python3 /work/kernels/moe-nvfp4/audit_cutlass_scales_cpu.py'
sg docker -c 'docker exec -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 qwen-prefill-dev python3 /work/kernels/moe-nvfp4/verify_cutlass_exact.py --preflight'
```

The audit first rejected the uncorrected candidate, then passed the exact
candidate. The verifier's CPU preflight constructs all three actual modular
kernels and verifies unchanged forward methods and b12x scale preparation.

Only when root schedules GPU work:

```bash
sg docker -c 'docker exec -e PYTHONDONTWRITEBYTECODE=1 qwen-prefill-dev python3 /work/kernels/moe-nvfp4/verify_cutlass_exact.py --gpu'
```

The verifier compares actual default `FlashInferExperts`, `shared`, and
`shared_exact` using real weights/routing and full `FusedMoEKernel.apply` calls.
It checks live GPU alphas, full-call CUDA-graph timings in both orders, and
small token counts 1/3/4/12/128/512/1023. Bitwise equality is required when
repeated reference/candidate outputs are stable; otherwise error is bounded by
observed reduction noise. At 1024 tokens it compares exact versus unchanged
shared b12x instead. One process shares the CUTLASS tactic cache across variants.
A separately scheduled explicit-greedy (`--temperature 0`) short-decode A/B
controls sampling but does not fix numerical or scheduling variation. The
executed results are recorded below.


## Executed GPU validation (2026-10-28 local)

[`results/cutlass_exact_gpu.jsonl`](results/cutlass_exact_gpu.jsonl) and
[compiler/runtime log](results/cutlass_exact_gpu.stderr) record a successful
exit with layers 0/19, tokens 1/3/4/12/128/512/1023/1024, three routing patterns,
and five graph replays per pattern. All **48 comparisons passed** the stated
criterion. Only one comparison had both outputs repeat-stable; it matched
bitwise. Other comparisons are bounded by measured nondeterministic reduction
noise, so this result does not claim universal bitwise output equality.

All four loaded alpha comparisons match the actual default exactly. Both
MMA-layout scale tensors alias their existing swizzled scale storage on both
layers and both hybrid variants; they allocate no extra scale copies.
Thirty-two interleaved timing records cover both orders of the three variants.
These are small-layer diagnostics, not evidence that alpha correction improves
serving throughput. The subsequent serving and quality diagnostics below
retain separate identities from the older `shared` runs.


## Executed serving and repeated diagnostics

`exact_tasks_1` completed all 17 serving workloads and the quality probes,
then greedy decode and repeated NLL. `next_diagnostic_1` completed greedy
decode and repeated NLL only. Both exited zero and passed source snapshot
verification. This difference in prior workloads limits causal comparisons
even though each measured phase uses its own warmup and cache salt.

The [full serving table](../../results/qwen_sequential/exact_vs_next_tasks_1.serving.md)
compares exact with the **earlier** `next_tasks_1`, at utilization 0.78:
8k/16k/32k TTFT p50 changes 191.6→187.3 / 384.5→363.3 / 866.3→809.3 ms
(−2.3%/−5.5%/−6.6%), while 2k worsens 66.2→74.1 ms (+11.9%). Seven workload
throughputs rise 0.5–9.0%, but workload TTFT and sampled decode are mixed.
The earlier shared run's larger prefill gains do not reproduce at that size.
The [runtime audit](../../results/qwen_sequential/prefill_runtime_cpu_audit.md)
records matching launch config and generated code, but different long-prefill
dense FP4 tactics and native FP8 quantizer configurations. These are concrete
confounds; their latency contribution has not been measured.

The [explicit-greedy comparison](../../results/qwen_sequential/exact_vs_next_greedy.md)
with `next_diagnostic_1` changes output rate by −0.51%/+1.41%/+0.78%/−0.73%
at c=1/4/16/64, near neutral in this pair. MTP acceptance and generated paths
still differ. This does not attribute the historical decode loss to alpha
rounding or prove a throughput benefit from the correction.

The [cold NLL comparison](../../results/qwen_sequential/exact_vs_next_repeated_nll.txt)
([JSON with hashes](../../results/qwen_sequential/exact_vs_next_repeated_nll.json))
contains five rounds of ten identical prompts: 50 valid calls per backend,
temperature zero, prompt-logprob cache reads skipped and fresh per-request
cache salts. Selected examples show the mean and observed repeat range:

| prompt | next mean [min, max] | exact mean [min, max] | mean delta |
|---|---|---|---:|
| licenses, 30k chars | 0.093348 [0.085723, 0.099457] | 0.095559 [0.088081, 0.098921] | +0.002212 |
| Python stdlib, 30k chars | 0.528470 [0.525201, 0.530838] | 0.531588 [0.520988, 0.535929] | +0.003118 |
| logprobs, 3,443 tokens | 0.363411 [0.344979, 0.385570] | 0.360551 [0.291359, 0.418547] | -0.002860 |
| logprobs, 7,604 tokens | 0.578589 [0.573978, 0.585333] | 0.573526 [0.560142, 0.585280] | -0.005063 |

All ten mean deltas lie between −0.008684 and +0.003118, and each absolute
delta is below 0.21 times the larger within-backend range. The earlier large
NLL deterioration is not reproduced in these repeats. Several means still
increase; observed ranges are not confidence intervals or an acceptance
threshold. The separate one-shot licenses NLL remains worse than the older
baseline (0.100423→0.105741). These probes do not establish broad quality
equivalence, and the quality change is not attributed to the two load-time
alpha expressions.

`shared_exact/` is the experimental parent for incremental stack testing.
It remains opt-in outside `CONFIG=next`; final composition and agentic
evaluation remain pending.
