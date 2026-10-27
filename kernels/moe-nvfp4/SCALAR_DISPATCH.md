# Scalar FC1-scale dispatch experiment

Status: **GPU correctness, microbenchmark and one controlled serving pair completed**.
Scalar remains an experimental composition candidate; quality equivalence and
patch-only attribution are not established. Results are below. Neither the `shared/` backend nor the existing
workspace verifier/harness nor installed container sources were changed.

## Change and limits

`scalar/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py`
is a separate copy of `shared_exact/`, retaining the corrected CUTLASS alpha
arithmetic from the original processed activation scales. Set
`VLLM_B12X_SCALAR_INPUT_SCALE=1` **before importing this overlay** to enable
its single added weight-loading hook. `VLLM_B12X_FIX_SCALES=1` is also required.
The existing shared-workspace flag is independent and is enabled in the
comparison script for both variants.

The hook replaces an exactly uniform, finite, positive FC1 input-scale
vector `[256]` with an independent `[1]` tensor. All nonuniform values remain
unchanged, including their original tensor identity. It is restricted to
BF16, SiLU, E=256, H=2048, I=512, top-k=8, local E=global E, no EP, and the
existing hybrid threshold >=1024 tokens. Both weight alphas and FC2 scales
retain their existing representation. Small batches keep the CUTLASS path.

Uniformity is checked with a 1 KiB CPU copy once per eligible layer during
weight preparation. This is a **load-time synchronization**, not a forward
or CUDA-graph replay synchronization. `apply`, wrapper ownership, and CUTLASS
execution are identical to `shared_exact/`; the CPU preflight compares their ASTs.

## Installed-source findings

Sources were read from `qwen-prefill-dev` at
`/usr/local/lib/python3.12/dist-packages/flashinfer/` on 2026-10-27.
Line numbers below refer to this installed copy, not upstream latest.

- `fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py:2437` remembers
  `input_gs.numel() == 1` before expansion and passes the result to
  `_get_dynamic_kernel` at line 2458. The compiler gates this on FP4 and
  top-k<=32 at lines 2192-2197; the shared bit participates in its cache key.
- `.../moe_dynamic_kernel.py` selects `MoEGatedDynamicKernel` for gated
  activations, SF vector=16, tile=(128,128), H<=16384, I<=512, and top-k<=16
  for the nonshared path (<=32 shared). Qwen's H2048/I512/top-k8/SiLU matches
  **both** branches, provided the workspace tile M is 128. The verifier checks
  that actual workspace condition. Default NVFP4 dynamic selection starts
  above 1024 routed rows (128 tokens at top-k8); this experiment's existing
  vLLM hybrid threshold remains 1024 **tokens**, well inside that band.
- `.../_moe_dynamic/gated.py:3448` already compares selected route scales and
  quantizes each 16-value input block once when they are equal. Scalar dispatch
  does **not** remove eight quantizations from the current uniform baseline.
  It hoists scale loading, skips per-block equality checks, caches route
  addresses, and unrolls top-k8 stores (lines 3122, 3232-3303). It still writes
  each route's packed data and block scales.
- `fused_moe/cute_dsl/b12x_moe.py:665-679` broadcasts
  `w1_alpha[E] * input_global_scale[1]` into the same `[E]` folded alpha as
  `w1_alpha[E] * uniform_input_global_scale[E]`, caching by tensor pointers
  and versions. `launch_sm120_moe` skips its own fold when wrapper weight
  views are supplied (dispatch lines 3479-3486). The overlay does not fold
  alpha itself, avoiding double scaling. FC2 remains `w2_s2 * a2`.
- `flashinfer/cute_dsl/fp4_common.py:2640-2649` uses
  `block_scale = amax / (6 * global_scale)` and
  `payload = x / (quantized_block_scale * global_scale)`. This is the existing
  dequant-style a13 convention; scalar versus repeated vector changes no value.
- Direct-micro's own scale contract is multiplier form. Dispatch lines
  1627-1644 can invert reciprocal scales, but only when
  `input_scales_are_reciprocal=True`. `B12xMoEWrapper.run` does not pass that
  argument, leaving the launch default False. The existence of this bridge
  does **not** establish safety for sending this backend's static scales to
  direct-micro. The experiment preserves the conservative hybrid gate.
- EP remains unsupported: the wrapper rejects local E != global E, and the
  dynamic kernel indexes routing state by global expert IDs. Scalar dispatch
  is not an EP-enablement change.

**Cost to measure:** `_expand_to_experts` (dispatch line 1494) performs
`scalar.expand(E).contiguous()` on every launch. The scalar case therefore
adds a small device broadcast/copy and an eager temporary allocation; CUDA
graph capture retains its allocation and replays the copy. Folded alphas are
cached, but this expansion is not. Any kernel savings compete with this cost.
No speedup or net allocation improvement is claimed. Avoid changing FlashInfer
internals until the simple full-call experiment provides evidence.

Source SHA256 values:

| File under FlashInfer | SHA256 |
| --- | --- |
| `fused_moe/cute_dsl/b12x_moe.py` | `c09065511e93c0ccf40e4d99f9215157aac9c26e0796ab41d1c52d396b45a2ff` |
| `fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py` | `c518e65d6bfd7f08db1e5261e20795fd020e82e681699729171bc2fd5331239a` |
| `fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py` | `7532e1338df8e80c4b8f0bd6473c2ffeda9a5028018c77775c4e9fe2fa47b072` |
| `fused_moe/cute_dsl/blackwell_sm12x/_moe_dynamic/gated.py` | `993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445` |

## Validation and later run

`verify_scalar_dispatch.py --preflight` runs the actual hook extracted from
its AST on CPU tensors. It checks 24 uniform/invalid/nonuniform/configuration/
flag cases, exact scalar/vector alpha-fold equivalence, and unchanged forward
methods and exact-alpha scale preparation. A CPU fixture also checks that the
original input scales survive without a reciprocal round trip. The GPU fixture
retains those scales explicitly, and both variants must have identical CUTLASS
alphas. The shared baseline and its older verifier remain unchanged.
The initial preflight failed with
`scalar selection not implemented`; after implementation all 24 passed with
`cuda_initialized=false`. Both Python files passed source compilation.

Repeat the CPU-only check:

```bash
sg docker -c 'docker exec -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 qwen-prefill-dev python3 /work/kernels/moe-nvfp4/verify_scalar_dispatch.py --preflight'
```

To reproduce the scheduled GPU mode on an idle GPU:

```bash
sg docker -c 'docker exec -e PYTHONDONTWRITEBYTECODE=1 qwen-prefill-dev python3 /work/kernels/moe-nvfp4/verify_scalar_dispatch.py --gpu --tokens 1023 1024 8192 --layers 0 19'
```

This loads each overlay directly from `/work` without mounting over installed
sources. It uses real Qwen weights/hidden states, checks actual dynamic shared
flags during warmup/capture, verifies the gated dispatch geometry and alpha
shape, and compares captured full `apply()` calls (including scale expansion,
ID masking, and output copy). It interleaves timing in both orders and checks
real/hot/masked/shifted routing over replay. A deliberately nonuniform FC1-scale
case confirms the generic path remains selected. That nonuniform case is a
representation-preservation comparison against `shared_exact/`, not a new checkpoint
accuracy claim. Tolerance is relative L2 <= max(0.002, 3 * baseline-repeat noise).
Independent source review found no substantive defect. Its coverage finding
was addressed: the nonuniform eight-hot fixture selects expert 17 explicitly
alongside experts 0–6, guaranteeing mixed selected scales are exercised.
Negative-ID tests run only in the b12x band; the CUTLASS boundary case tests its
unchanged route contract. GPU validation must pass and show a repeatable full
call benefit before adoption. Executed microbenchmark results are recorded below.


## Executed GPU result (2026-10-28 local)

Both scheduled processes exited 0. Together they passed **224 output/replay
comparisons**, 40 dispatch checks, and recorded 40 timing rows, using actual
checkpoint layers 0/19 with real captured hidden states. Uniform scales select
the shared-input dynamic implementation; deliberately nonuniform scales retain
the generic implementation. Negative routes are tested only above the hybrid
threshold. The 1023-token case remains CUTLASS. All-masked output is zero.
Numerical tolerance is still bounded by baseline repeat noise, not bitwise
agreement or a full-model quality guarantee.

Times below average the medians from A/B and B/A orders; they include scale
broadcast, masking and output copy. Negative duration change means faster.

| layer | tokens | shared_exact us | scalar us | duration change |
|---:|---:|---:|---:|---:|
| 0 | 1023 | 355.71 | 356.82 | +0.31% |
| 0 | 1024 | 333.95 | 330.80 | -0.94% |
| 0 | 4096 | 455.34 | 440.70 | -3.22% |
| 0 | 8192 | 629.10 | 617.01 | -1.92% |
| 0 | 16384 | 1162.62 | 1157.28 | -0.46% |
| 19 | 1023 | 321.41 | 323.84 | +0.76% |
| 19 | 1024 | 316.53 | 311.33 | -1.64% |
| 19 | 4096 | 449.88 | 434.62 | -3.39% |
| 19 | 8192 | 637.98 | 626.48 | -1.80% |
| 19 | 16384 | 1156.29 | 1161.62 | +0.46% |

The 4k saving is 14–16 us/layer and the 8k saving is 11–12 us/layer. If those
shapes occurred in all 40 MoE layers, the arithmetic sum would be roughly
0.5–0.6 ms per forward, not a measured server saving. At 16k the result is
neutral within roughly ±0.5%, and the unchanged CUTLASS boundary shows small
variation too. This is enough to test the scalar path in serving, not to claim
a universal 2–3% model speedup. Final composition remains undecided.

Raw results: [initial](results/scalar_exact_gpu.jsonl),
[4k/16k extension](results/scalar_exact_prefill_gpu.jsonl), with matching
`.stderr` compiler logs. The first run uses two correctness replays and 15
timing repetitions/order; the extension uses three and 30, respectively.

```bash
python3 verify_scalar_dispatch.py --gpu --tokens 1023 1024 8192 --layers 0 19
python3 verify_scalar_dispatch.py --gpu --tokens 4096 16384 --layers 0 19 --replays 3 --repeats 30
```

## All-layer checkpoint eligibility on CPU

The [all-40-layer CPU audit](results/scalar_all_layers_eligibility_cpu.json),
produced by [audit_scalar_eligibility_cpu.py](audit_scalar_eligibility_cpu.py),
executes the installed activation-scale preparation expressions and the actual
scalar-selection hook against all 30,720 cached gate/up/down input-scale tensors.
All **40/40 target MoE layers** have finite, positive, exactly uniform prepared
FC1 scales and select an independent `[1]` scalar preserving every vector value,
under the recorded TP1/no-EP/BF16/static-scale/flags-enabled/threshold-1024 assumptions.
The report records each layer's guards, exact input/source hashes, checkpoint
revision, and checked packed-weight header shapes. CUDA remained uninitialized.
This establishes checkpoint eligibility; it does not observe live per-layer GPU
dispatch or extend the layers-0/19 microbenchmark into an all-layer speed claim.

## Matched vLLM serving pair

`exact_matched_3` and `scalar_matched_3` both completed exit 0, sequentially on the
otherwise idle GPU, using the same full individual suite, explicit greedy decode,
and three cold NLL rounds. Source snapshots verify 23/23 files each. Both preserve
all 132 reference FlashInfer entries through preload, graph capture and graceful
shutdown. Inductor is **not pinned**: four of 13 saved tuning records differ. These records alone do not establish
different final runtime choices.
The changes below describe the measured configurations, not a causal scalar-only
speedup or a confidence interval.

| cold prefill tokens | shared_exact TTFT p50 ms | scalar TTFT p50 ms | change |
|---:|---:|---:|---:|
| 2048 | 68.52 | 68.19 | -0.5% |
| 4096 | 107.51 | 100.54 | -6.5% |
| 5000 | 120.86 | 115.44 | -4.5% |
| 8192 | 185.15 | 164.08 | -11.4% |
| 16384 | 384.57 | 343.63 | -10.6% |
| 32768 | 838.71 | 811.50 | -3.2% |

Seven workload throughputs change -5.3% to +1.5%; the largest regression is
long-doc, where MTP acceptance also falls 38.3%→32.4%. Explicit greedy decode
output rate changes +0.53%/+4.63%/-1.91%/+1.99% at c=1/4/16/64. Sampling at
temperature 0 does not eliminate numerical, MTP trajectory or scheduling changes.
Both allocate 45.50 GiB KV, 3,437,190 KV tokens and 3.45 GiB final CUDA graph pool.

Both pass 20/20 needle checks. Three cold repeats on each of 10 fixed prompts give
NLL mean deltas from -0.042482 to +0.019169. The py_email-30000 mean worsens
0.794235→0.802070; its three-repeat ranges do not overlap
([0.790536, 0.797625] vs [0.798475, 0.805086]). Retain this quality flag;
these few diagnostic prompts do not establish either broad degradation or
quality equivalence. Checkpoint prefix-reuse TV(top20) mean is 0.1400/0.1367
and top1 agreement 10/12 vs 11/12, also diagnostic only.

The server's 8k/16k gains are much larger than this patch's full-call micro gain,
especially the neutral 16k micro result. The [CPU compiler audit](../../results/qwen_sequential/scalar_inductor_runtime_cpu_audit.md)
links all 13 matching wrappers on each side to saved model artifacts: ten GDN
wrappers contain identical single candidates, while three artifacts retain
multiple candidates with unresolved final in-memory choices. Candidate sets and
executable ELF sections match; this does not establish every runtime launch or
loadable datum was identical. A compiler cause remains unproven. The fresh
controlled `next` baseline and incremental `scalar_gdn` measurement have completed;
see the [full composition results](../gdn-norm-fp8/results/COMPOSED_SERVING.md).
Scalar remains the experimental parent while a focused reversed-order pair
checks the incremental GDN tradeoffs before final selection.
The default `next` configuration is unchanged. Agentic remains a final composed
comparison and has not been run for this individual patch.

Reports: [17 serving tasks](../../results/qwen_sequential/scalar_vs_exact_matched_3.serving.md),
[greedy decode](../../results/qwen_sequential/scalar_vs_exact_matched_3.greedy.md),
[cold NLL repeats](../../results/qwen_sequential/scalar_vs_exact_matched_3.nll.txt).
Raw directories are `results/qwen_sequential/{exact,scalar}_matched_3/`.

```sh
FI_CONTROL=1 GREEDY_DECODE=1 NLL_ROUNDS=3 bash bench/qwen_experiment.sh exact_matched_3 shared_exact individual
FI_CONTROL=1 GREEDY_DECODE=1 NLL_ROUNDS=3 bash bench/qwen_experiment.sh scalar_matched_3 scalar individual
```

Use fresh tags when reproducing; existing evidence directories are protected.
