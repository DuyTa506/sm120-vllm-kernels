# DeepGEMM scheduling ideas and native NVFP4 — 2026-10-28

Keep the experiment at the MoE-call boundary: routing, quantization, GEMMs,
activation and output combine. The useful question is which concrete scheduling
or pipeline change beats the current Qwen implementation. Persistence alone is
already present in the installed B12x implementation.

## DeepGEMM branch and precision distinction

The official [`nv_dev` tree at b64107f](https://github.com/deepseek-ai/DeepGEMM/tree/b64107f2b9599ca76445b7f62eedf66bae1d095b)
contains native SM120 GEMM, scheduler, MMA and test files. The open
[SM120 issue](https://github.com/deepseek-ai/DeepGEMM/issues/236) is not proof that
no SM120 implementation exists: its discussion also identifies the separate
`nv_dev` integration. Our experiments already executed a native SM120 FP8
rowwise adapter, including live vLLM graph nodes; its serving gain remains
unselected. This is separate from a checkpoint-compatible NVFP4 MoE implementation.

The examined SM120 FP4 path uses block-32 UE8M0 (`scale_vec::2X`). Our NVFP4
experts need block-16 E4M3 (`scale_vec::4X`), correct global factors and rounding.
Porting the scheduler does not require inheriting that FP4 representation.
SM120's native register MMA path also differs from SM100 TMEM/tcgen05.
[Colfax's implementation tutorial](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/)
and the [CUTLASS SM120 example](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_cooperative.py)
provide relevant load/compute primitives.

## What the current stack already does

| Proposed experiment | Existing mechanism | Difference worth measuring |
|---|---|---|
| E0: baseline | Scalar wrapper uses CUTLASS below M1024 and B12x above, with the documented conditions | Record actual dispatch; explicitly enabled FI-auto/direct are additional comparators, not assumed serving baselines. |
| E1: persistent grouped scheduling | Native dynamic B12x has persistent task claims and compact descriptors | Queue policy, task size and weight reuse. Our long-first permutation improved a hot-layer case but has no demonstrated serving win. |
| E2: masked grouping | Device-side route counts, valid rows, inactive-M16 MMA predicates and fixed-capacity graph execution already exist | Target remaining operand loads or descriptor overhead; reduced reserved rows alone do not establish less MMA work. |
| E3: shape-specific tile selection | Installed functional selector supports M16/32/64/128; the graph wrapper retains maximum-capacity geometry | The tested workspace bands recover smaller tiles but regress M1024/M2048. Use a measured cost model; do not select by padding fraction alone. |
| E4: warp/TMA pipeline | Native MMA and pipelining are already part of the implementations | Compare concrete stage count, copy mechanism, register allocation and occupancy for FC1 K2048 versus FC2 K512. |
| E5/E6: activation fusion / whole MoE fusion | B12x already combines expert computation and activation work | Initial native split loses about 17.6% at M1024 and 47.4% at M4096 in complete-call tests; a different pipeline needs new evidence. |
| E7: hybrid dispatch | Current wrapper already separates small and large M | Re-evaluate the crossover using full-call and serving measurements; multiple implementations can win in different ranges. |

For E3, installed `_select_dynamic_tile_m` changes tiles at mean routed rows
15, 48 and 96. With E256/top8 these correspond to token-count boundaries
480, 1536 and 3072 inside that selector; the outer backend dispatch still applies.
Crucially, the installed graph wrapper allocates its dynamic workspace once from
`max_num_tokens * top_k`. Dispatch retains an explicitly supplied workspace's
tile geometry. Thus the functional selector does not imply per-call adaptation
in our serving wrapper. The separate immutable-band adapter has now completed
a captured L19 full-call screen: M1024/tile32 regresses about 1.9%,
M2048/tile64 about 27.1%, and unchanged M4096/tile128 stays within 0.32%.
Its three bands allocate 1,400,907,776 bytes versus 607,809,536 bytes for the parent.
Six graph transitions and exploratory diagnostics pass, but the policy remains
unselected. See the [report](../results/round2_snapshot/workspace_bands_report.json)
and [summary](../results/round2_snapshot/workspace_bands_summary.json).

Saved L19/M2048 routing contains 16,384 expert assignments. M128 reserves 35,072
rows in 274 expert tiles; M64 would reserve 24,896 rows in 389 tiles. That raises
useful-row fraction from 46.7% to 65.8%, but also increases tile count by 42.0%.
This is a storage/padding calculation, **not a compute-saving estimate**. The
[actual installed M128 source](../results/round2_snapshot/workspace_band_source_analysis.md)
already skips inactive M16 MMA fragments in both FC1 and FC2, and honors valid
rows during quantization/scatter. Smaller tiles can add task and weight-loading
overhead without removing the presumed padded MMA work. The largest expert
receives 1,137 rows despite a mean of 64, illustrating why the mean alone is
insufficient. See the [saved route analysis](../results/round2_snapshot/workspace_band_geometry.json).

A distinct remaining hypothesis is to skip shared-to-register operand loads for
warps with no active M16 fragment. The source currently guards the MMA but leaves
some A/B/SFA/SFB loads outside that predicate. Such a change must keep every
pipeline wait/release, named barrier and quantization/scatter duty intact, use a
warp-uniform condition, and inspect compiled register/load effects. It is not
additional MMA elimination or a demonstrated speedup.

The [DeepGEMM MegaMoE heuristic](https://github.com/deepseek-ai/DeepGEMM/blob/5081f8513af57e61696de4f4f04b0e707ffb503e/csrc/jit_kernels/heuristics/mega_moe.hpp)
considers expected load plus a square-root margin and a weight-reload penalty.
Borrow that cost tradeoff, not its SM100 block sizes or numerical thresholds.
Keep expert counts and tile publication on the GPU; calibrate static choices
from actual per-layer route histograms without reading counts back each forward.

## New source: sm120_nvfp4_ops

Reviewed [commit 6e0a06e](https://github.com/xinyang-zhou/sm120_nvfp4_ops/tree/6e0a06eb8321d6c9c7713446a60560b3cff92350).
Its [grouped kernel](https://github.com/xinyang-zhou/sm120_nvfp4_ops/blob/6e0a06eb8321d6c9c7713446a60560b3cff92350/src/grouped_gemm/kernels.cuh)
publishes invocation-specific TMA descriptors on the GPU, computes expert tile
metadata, then persistently strides through a flattened tile domain. It uses
384 threads, including 256 compute threads, with a separate loading group.
This is a concrete pipeline and descriptor-management reference.

The [dispatcher](https://github.com/xinyang-zhou/sm120_nvfp4_ops/blob/6e0a06eb8321d6c9c7713446a60560b3cff92350/src/grouped_gemm/grouped_gemm.cu)
currently fixes **M128/N128/K128 and three stages**; its average-rows argument is
unused. Every invocation refreshes descriptors and tile metadata. These costs
must be counted, and small-Qwen shapes need their own policy. The published
[grouped results](https://github.com/xinyang-zhou/sm120_nvfp4_ops/blob/6e0a06eb8321d6c9c7713446a60560b3cff92350/docs/PERFORMANCE.md)
use M16/N4096/K8192 and a cuBLASLt loop, on RTX 5090 by the stated default
methodology. They do not measure an advantage over Qwen's fused MoE backend.

Its [MoE API](https://github.com/xinyang-zhou/sm120_nvfp4_ops/blob/6e0a06eb8321d6c9c7713446a60560b3cff92350/src/fused_moe/fused_moe.cu)
orchestrates separate gather, FC1, activation/quantization, FC2 and reduction
operations. The [activation code](https://github.com/xinyang-zhou/sm120_nvfp4_ops/blob/6e0a06eb8321d6c9c7713446a60560b3cff92350/src/fused_moe/activation.cu)
consumes FP16 intermediates and forms local E4M3 scales without our calibrated
global factors. Keep its scheduler/pipeline ideas, but adapt and verify the
checkpoint arithmetic before running its MoE as a replacement. A native NVFP4
instruction alone does not establish the full numerical contract.

## Concrete work now

- **Prefill:** the isolated native NVFP4 split adapter passes full-domain
  arithmetic and changed-route graph checks, including empty then restored
  inputs, but loses complete-call timing at both checked shapes. L19/M1024 is
  approximately 369 µs versus 314 µs scalar; M4096 is 640 µs versus 434 µs. The
  [timing summary](../results/round2_snapshot/b12x_split_initial_summary.json)
  retains raw ranges, reversed orders and a repeat. It is unselected.
  Separate component graphs diagnose publisher and phase costs under different
  cache conditions; their times must not be summed as a full-call result.
  M64 compute tiles do not validate the upstream M64 source-domain variant;
  its source warns against that multi-tile path.
- **Decode:** the isolated DP4A expert-reuse port passes the first M1 GPU
  numerical screen: input quantized bytes match vLLM, FC1/FC2 agree with FP32
  accumulation of the same quantized operands to about 1e-7 relative L2, and
  graph/eager and instrumented/normal outputs match. The M8 follow-up also
  exercises real shared experts and all eight tokens sharing eight experts;
  reuse on/off outputs remain identical. Timing must use the strongest valid
  comparator: historical M1 FI calls still converted fixed scales every time.
  The [M3 batched screen](../results/round2_snapshot/dp4a_micro_m3_batched.json)
  measures calibrated preplanned FI direct at about 24.6 µs, DP4A reuse-on at
  about 32–35 µs, and current scalar/CUTLASS at about 49 µs. DP4A is not selected
  from that result. The preplanned direct ABI takes multiplier scales; blindly
  forcing the existing wrapper into direct-micro would violate its calibrated
  convention. No vLLM serving, concurrency or model-quality equivalence follows.
- **Scheduler:** retain long-first as an unselected measured case. Use route
  distributions to propose a different cost-aware policy rather than repeating
  the same permutation under a new name.

Marlin remains a useful same-checkpoint control. NVIDIA's
[GB200 result](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/megatron_bridge/tutorials/Qwen3.6-35B-A3B/README.md)
uses four GB200s, vLLM 0.28 and TP4+EP; it cannot eliminate W4A16 as a possible
small-M SM120 winner. Native W4A4 remains the primary prefill direction.
Humming's promising M1024 layer timings also require actual serving-bucket coverage and memory validation:
the CPU adapter estimate adds 16.875 GiB of duplicate weights/local scales, and
standard serving tasks have not yet proved they reach that bucket. The measured
default and serving chart remain unchanged.

[Pinned source audit](../results/external_review/scheduler_source_audit.json) ·
[Broader external review](EXTERNAL_KERNEL_REVIEW.md) ·
[Measured results and limits](PROGRESS.md).
