# External SM120 kernel review — 2026-10-28

The comparison target stays `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast`, vLLM 0.32.0,
one RTX PRO 6000, with the existing attention settings and MTP2. A previous loss
rejects that measured implementation and shape, not the whole optimization idea.
This review identifies mechanisms to test; it adds no speedup to the progress chart.

## B12x is already in FlashInfer, but the implementations differ

Our image uses FlashInfer **0.7.0.post1** and its
`fused_moe/cute_dsl/blackwell_sm12x` implementation through vLLM's B12x adapter.
The earlier vLLM integration is [PR 40082](https://github.com/vllm-project/vllm/pull/40082).
The later [FlashInfer PR 5767](https://github.com/flashinfer-ai/flashinfer/pull/5767),
merged October 6, imports the larger B12x package under
`flashinfer/experimental/b12x`; the [original repository](https://github.com/local-inference-lab/b12x)
is archived. Installing the package and selecting its prepared execution API are
separate operations. Our installed image has **no experimental/b12x directory**.

We inspected both the migration commit `fc8fdc17f702371302a0ea780c604b099b927196`
and October 28 main `fc4df7ee1a1edbd713ff1438c25f8de6b3e5d3af`.
[Source hashes and runtime evidence](../results/external_review/b12x_source_audit.json)
record the precise comparison. The two NVFP4 phase kernels and weight planner
have identical Python ASTs between these commits despite formatting changes.
This does not imply that every file in the package is unchanged.

## Sources and transferable mechanisms

| Source reviewed | Relevant mechanism | Relation to our work / next check |
|---|---|---|
| [B12x / FlashInfer](https://github.com/flashinfer-ai/flashinfer/tree/fc4df7ee1a1edbd713ff1438c25f8de6b3e5d3af/flashinfer/experimental/b12x/moe) | Prepared execution, separate NVFP4 prefill phases, decode policies | Existing stable backend is only a partial overlap. Compare actual code and dispatch; prioritize the prefill phases below. |
| [lna-lab](https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm/tree/acabe06f6eea9e11dfb58dd43e400782752835a5) | SM120 grouped tile generation, shared-memory limits, backend admission | Basic admission already works here. Audit legal tile policies still missing from our candidates before skipping the tuning mechanism. The reported 175 tok/s uses another checkpoint and is not a matched performance comparison. |
| [CUTLASS 79d](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu) | Grouped NVFP4/TMA implementation and tile policies | Inspect per-expert padding, tile-K and final shared-memory use against real route histograms. Example **87a is blockwise FP8**; 79a/79b/79d are the relevant NVFP4 examples. |
| [CUTLASS issue 3096](https://github.com/NVIDIA/cutlass/issues/3096) | Historical SM120 correctness and shared-memory investigation | Useful failure modes to verify, not proof our current backend fails. The report used vLLM 0.17, FlashInfer 0.6.5 and a different, multi-GPU model; it was closed April 14. |
| [vLLM NVFP4 oracle](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py) and [Qwen recipe](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.6-35B-A3B.yaml) | Marlin W4A16 execution of NVFP4 weights | Installed source supports testing our same compressed-tensors checkpoint with Marlin. Keep the target/backend change separate from checkpoint, attention and speculative-decoding changes. |
| [nickus/vllm-sm120](https://github.com/nickus/vllm-sm120) | Older backend-selection overrides enabling native CUTLASS fallback | Targets vLLM 0.20. Skip transplanting obsolete rejection rules; retain the lesson to verify which kernel actually executes. Its benchmark changes checkpoint/quantization and cannot settle our backend choice. |
| [fastkernels](https://github.com/suryavanshi/fastkernels/tree/40cbe1476400c1b8d270c14872c1b3baa9664fe6) | Routed/shared down-projection combination and staged Qwen operators | Study the output boundary and accumulation, then adapt NVFP4 scales and stream dependencies. Its 1052 tok/s megakernel result is a 0.8B BF16 model on B200, not the 35B NVFP4 serving path. |
| [FlashInfer issue 5561](https://github.com/flashinfer-ai/flashinfer/issues/5561) and [public CUDA kernel](https://github.com/SSHdotCodes/qwen-3.8-flash-next-pro6000/blob/8e5be4440eeea37d31994b0845d1344201de7741/serve/qwenfast/qf_moe.cu) | Expert deduplication across tokens, DP4A arithmetic, two launches and final reduction | Strong decode experiment. Port H2560/I640/top10 to H2048/I512/top8, and separate expert reuse from arithmetic and PDL effects. |
| [SGLang issue 33706](https://github.com/sgl-project/sglang/issues/33706) | Shared/routed expert fusion proposal | Proposed for blockwise FP8; closed for inactivity, not a demonstrated merged NVFP4 optimization. Our formats permit investigation, but distinct calibration and shared sigmoid gating must survive. |
| [NVIDIA NVFP4 discussion](https://forums.developer.nvidia.com/t/nvfp4-performance-update/367781) | SM12x architecture and integration context | Confirms why SM100 implementations cannot simply be reused. Performance must still be measured on our SM120 operands and serving path. |

## Priority experiments

### 1. Prefill: separate routing, FC1 and FC2

The experimental B12x [FC1](https://github.com/flashinfer-ai/flashinfer/blob/fc4df7ee1a1edbd713ff1438c25f8de6b3e5d3af/flashinfer/experimental/b12x/moe/_shared/kernels/nvfp4_phase1.py)
and [FC2](https://github.com/flashinfer-ai/flashinfer/blob/fc4df7ee1a1edbd713ff1438c25f8de6b3e5d3af/flashinfer/experimental/b12x/moe/_shared/kernels/nvfp4_phase2.py)
kernels separate GEMM resource use from the routing kernel. They use native SM120
NVFP4 MMA, M64×N128 compute tiles and fixed launch capacities. FC1 writes packed
NVFP4 intermediates and scales; FC2 consumes them. The hypothesis is better
occupancy despite more launches and intermediate traffic. Our stable implementation
does not contain this path; the earlier Humming result does not evaluate it.

The [admission policy](https://github.com/flashinfer-ai/flashinfer/blob/fc4df7ee1a1edbd713ff1438c25f8de6b3e5d3af/flashinfer/experimental/b12x/moe/fused_moe/_tuning.py)
requires the M128 source domain, SiLU, shared input scales and aligned dimensions.
It excludes deterministic output, dynamic down-scale, swap-AB and the ready-queue
publisher; those options cannot simply be stacked with this path. H2048/I512 meets the dimensional rules,
but actual scale and workspace contracts still need proof. The
[weight planner](https://github.com/flashinfer-ai/flashinfer/blob/fc4df7ee1a1edbd713ff1438c25f8de6b3e5d3af/flashinfer/experimental/b12x/moe/fused_moe/planning.py)
admits A4 via its ModelOpt representation; declaring a compressed-tensors source
does not automatically admit A4. Adapt our already prepared effective operands
with decoded-weight/scale checks, rather than relabeling the checkpoint.

Start with captured L19 M1024/M4096, then L0/L39. Compare the whole expert call
against current scalar, including routing, packing, intermediate storage and
reduction. Measure register/shared-memory occupancy and graph replay; only then
integrate a shape-specific vLLM adapter.

### 2. Decode: reuse expert weights across tokens

Our direct-micro already uses software FP4 dots and avoids a route prepass.
The public kernel's distinct opportunity is grouping tokens that chose the same
expert, so a CTA can reuse loaded weights. Preserve the model's expert selections;
this changes execution order and reuse, not learned routing.

Current automatic dispatch requires fewer than 32 routed pairs, so top-8 admits
M1–3 through that gate. Test **M1/2/3/4/8** with automatic and forced-direct
baselines. Include actual unique-expert counts, and compare reuse on/off while
holding DP4A arithmetic and PDL fixed. A warm layer loop alone cannot establish
DRAM bandwidth savings across 40 layers.

CPU port preparation found hard-coded loop bounds beyond H/I/topk, opposite
raw gate/up ordering, differing weight layouts, and process-static scratch that
needs replacement. An independent checker verifies FP4 integer products, scale
index coverage and route ownership; it is not CUDA correctness or performance.
The isolated port is in progress, with PDL initially off.

### 3. Same-checkpoint Marlin control

Test target MoE Marlin against B12x with the same snapshot and parent patches,
keeping the existing B12x draft backend and MTP2 explicit. Marlin changes the
activation arithmetic to W4A16; record quality and scale-preparation effects as
well as complete-call timing. CPU source selection and command comparisons are
prepared. Full GPU eligibility and correctness remain untested. Do not force
`--quantization modelopt_fp4` on this compressed-tensors checkpoint.

### 4. Shared-expert output boundary

First measure the shared branch and check existing compiler fusion. Preserve
shared/routed GEMM overlap, each branch's scales, the shared sigmoid gate and
top-8 normalization. Compare merging outputs against the current separate
execution, including synchronization. Full fusion as a 257th expert/ninth
contribution is a distinct, larger experiment; it need not precede the smaller
output-boundary test.

### 5. Revisit reductions and scheduling independently

The public decode kernel suggests a last-CTA finalizer; experimental B12x also
has optional deterministic output paths with restrictions. Compare actual
contribution values, accumulation order and scratch traffic before borrowing
either mechanism. Our task-order diagnostic already proves unchanged individual
contributions for its recorded case, while final BF16 scatter remains order
dependent. A useful queue permutation can therefore have both numerical and
scheduling effects.

The completed task-order serving pair remains unselected: candidate request
latency rises 11.24%, with two of ten common Inductor configurations differing.
Actual candidate execution and unchanged contributions do not establish a
serving benefit; [the measured status](PROGRESS.md) retains that distinction.

## Integration rule

Each candidate must first pass its operand/layout and graph-replay checks,
then complete-call timing and focused vLLM tasks. Compare actual executed
kernel symbols and cache identities, not only startup class names. Compatible
winners can compose through explicit shape dispatch. Keep the current progress
chart tied to completed serving results and run agentic only on the selected
final combination.
