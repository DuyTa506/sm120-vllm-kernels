# CUTLASS fallback alpha arithmetic audit

The isolated `shared_exact/` candidate corrects a real representation difference;
**GPU validation passed on layers 0 and 19**. It does not establish the cause of the
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
IDs and graph kernel sequences have not been compared.

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
autotuning, scheduling, and host variation remain confounds. The running task
matrix retains the original repository protocol.

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
After that passes, a separately scheduled explicit-greedy (`--temperature 0`)
short-decode A/B can isolate sampling variability. The executed results are recorded below.


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
serving throughput. The new `shared_exact` serving variant needs a fresh
serving/quality run before final composition; older `shared` results retain
their original identity.
