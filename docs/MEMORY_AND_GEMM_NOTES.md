# Memory hierarchy, GEMM throughput and prefill opportunities

Review checkpoint, 2026-10-28. The user requested notes and a push before reading
the work, then resuming optimization later. **Optimization is paused for that
review. No new GPU experiment or automatic follow-up is scheduled by this note.**

Read [HANDOFF.md](HANDOFF.md) for state, [PROGRESS.md](PROGRESS.md) for measured
serving results, and [PREFILL_COUNTER_FINDINGS.md](PREFILL_COUNTER_FINDINGS.md) for
the successful counter report. This note explains the concepts and remaining
questions; the candidate directions below are not measured improvements.

## Memory hierarchy: DRAM, SRAM and cache

The RTX PRO 6000 Blackwell Workstation Edition uses **96 GB GDDR7**, with published
bandwidth **1792 GB/s = 1.792 TB/s**. HBM on datacenter GPUs and GDDR7 are both
forms of **DRAM outside the GPU die**. Registers, cache and shared memory are
on-chip storage; the cache/shared-memory structures use SRAM. There is no second,
faster DRAM pool on this card that software can choose instead of GDDR7.
[NVIDIA datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/rtx-pro-6000-blackwell-workstation-edition/workstation-blackwell-rtx-pro-6000-workstation-edition-nvidia-us-3519208-web.pdf).

| Level | Role | How a GEMM benefits |
|---|---|---|
| GDDR7 / VRAM | Large storage for weights, activations, workspaces | Transfer fewer bytes, coalesce accesses, sustain bandwidth |
| L2 | Shared cache across SMs; this report identifies 128 MiB | Schedule consumers of the same data close enough to reuse it |
| L1 / shared memory per SM | Fast, small storage; shared is explicitly managed | Keep input tiles on-chip and reuse across warps/rows |
| Registers per thread | Operand fragments and accumulators | Reuse values near computation; avoid spills |
| Tensor Cores | Execute MMA instructions | Consume the supplied operands efficiently |

For SM120 warp-level MMA, operands/accumulators reside in registers. A common
path is global memory → shared memory → registers → MMA. Cache behavior depends
on the copy/load path; not every access traverses L1. Shared tiles and register
fragments are smaller pieces of matrices, not full expert weight sets.
[CUTLASS warp MMA dataflow](https://docs.nvidia.com/cutlass/4.8.0/media/docs/pythonDSL/guides/mma/wmma_programming.html).

Faster on-chip memory helps only when data is available there. A cold weight tile
must first arrive through GDDR7. Reusing it across many token rows amortizes that
transfer. A good cache hit rate is not sufficient by itself: prefetching may make
later accesses hit without reducing total DRAM traffic or elapsed time.
Shared-memory tiling avoids redundant global transfers, but consumes resources
and can introduce bank conflicts/barriers. [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html).

Bandwidth must be interpreted per level. L2 bandwidth is not additional bandwidth
that can be added to GDDR7's 1.792 TB/s. Nor should a per-SM shared-memory rate be
compared with a whole-device DRAM rate. Each level has its own traffic, throughput
and access constraints; removing one bottleneck can expose another.

## Peak FLOP/s, executed work and useful work

FLOP is an operation count; FLOP/s is a rate. For a dense GEMM
`[M,K] × [K,N]`, useful arithmetic is approximately **2MNK FLOP**, counting one
multiply-add as two operations. Actual execution also includes memory/layout,
scaling, bookkeeping, synchronization and possibly padded MMA work.
[NVIDIA GEMM background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/).

Three quantities must stay separate:

1. **Hardware peak:** depends on precision, accumulation, clock and dense versus
   structured-sparse execution. NVIDIA's headline FP4 AI TOPS uses sparsity.
   Choosing 8 of 256 MoE experts is not structured sparsity inside those GEMMs.
   [NVIDIA specification footnotes](https://www.nvidia.com/en-gb/products/workstations/professional-desktop-gpus/rtx-pro-6000/).
2. **Executed kernel work:** includes whatever tiles/instructions the binary
   actually issues. Allocating a tileM128 buffer does not prove all 128 rows run
   MMA: the current dynamic kernel already skips inactive M16 fragments.
3. **Useful model work:** valid routed rows only, often reported as `2MNK/time`.
   This excludes padding and the other work of the fused kernel.

Tensor-pipeline activity percentages are not useful FLOP/s divided by hardware
peak. The report's 20.87% tensor activity and roughly 165 TFLOP/s useful rate
therefore need not represent the same utilization percentage. A large dense
GEMM microbenchmark is also not the attainable peak for every small grouped GEMM.

## Why prefill MoE can remain bandwidth-heavy

For this model, E256/top8 gives the *mean* routed rows per expert:

```text
M_expert_mean = input_tokens × 8 / 256 = input_tokens / 32
1024 input tokens → 32 rows/expert on average
4096 input tokens → 128 rows/expert on average
```

Actual counts/skew must come from routing, not this expectation. Each expert has
FC1 `[M_e,2048] × [2048,1024]`, followed by activation and FC2
`[M_e,512] × [512,2048]`. A large total prefill does not imply a large M per GEMM.

The successful L19/M1024 diagnostic gives:

| Quantity | Value / meaning |
|---|---|
| Useful FC1+FC2 work | 51.540 GFLOP, excluding activation/quantization and padded work |
| Kernel duration | 311.680 µs; one selected native fused kernel |
| Aggregate DRAM traffic | Approximately 437.191 MB; reads/writes not separated |
| Useful arithmetic intensity versus DRAM | Approximately 117.89 FLOP/byte |
| Useful rate | Approximately 165.36 TFLOP/s |
| Illustrative rate at published 1.792 TB/s, unchanged intensity | Approximately 211.26 TFLOP/s |

These calculations use the [published summary](../results/prefill_counter_study_2/summary.json).
The last row is a simplified constant-traffic roofline estimate, not a measured
result or an independent proof of the limiter. The actual counter evidence is
high DRAM traffic rate relative to the reference ceiling, alongside much lower
compute/L2 throughput. Stall and achieved-occupancy measurements remain absent.

## How much can cache plausibly save here?

Packed FP4 weights for all 256 routed experts total **384 MiB**; E4M3 block scales
add **48 MiB**, giving **432 MiB** before other metadata. This exceeds 128 MiB L2.
One expert's packed weights plus block scales are **1.6875 MiB**, so reuse of a
smaller active working set can still fit even when the whole layer cannot.

However, the measured 437 MB traffic is already close to the 453 MB weight-plus-
scale volume for the whole layer. If all experts are touched and each weight is
nearly streamed once, there may be little repeated DRAM weight traffic to remove.
That is a hypothesis requiring byte attribution, not a proven compulsory-traffic
lower bound: cache residency, expert activity, activations and writes matter.
L2's 60.4% sector hit rate is not the fraction of expert weights retained.

At unchanged traffic, the earlier calculation against **1.79 TB/s** gives an ideal
244.24 µs versus 311.68 µs, or about **21.64% less time**. Using the precise published
1.792 TB/s changes that slightly, to about 243.97 µs. This is neither guaranteed
headroom nor an upper limit if traffic itself is reduced. It applies to the one
kernel, not whole-model TTFT or serving throughput.

## Candidate directions and the evidence each needs

| Direction | Potential mechanism | Question before implementation |
|---|---|---|
| Weight/scale reuse across tiles | Reduce repeated fetches through L2/shared | How often is each expert/tile really fetched? Grouping already exists. |
| Working-set-aware task order | Keep shared consumers close, reduce eviction | Does reuse improve without idle SMs, queue overhead or worse tails? Prior task-order trial has no serving win. |
| Reuse at larger M / batching | Amortize weight transfers over more token rows | Which serving M buckets occur, and what are TTFT, scratch and register costs? |
| Load/layout/pipeline improvements | Increase attained bandwidth or hide load latency | Are accesses inefficient, or is the kernel already near attainable sustained bandwidth? |
| Remaining intermediate traffic | Avoid unnecessary global writes/reads | What buffers still matter? Native MoE is already fused; generic fusion is not a new result. |

Cache policies/prefetch can influence residency, not guarantee that an entire
layer stays cached. A smaller tile can also increase weight reloads/task count;
more shared memory or registers can reduce residency or spill. The earlier
negative tile/load-guard variants remain excluded.

The next narrow investigation should compare **actual expert counts, DRAM reads
and writes, tile reuse, and representative cache conditions at M1024/M4096**,
then connect those costs to the existing serving timeline. Warm single-layer
replay and cold/40-layer execution can have different cache behavior. Preserve
router selections and probabilities; changing which experts the model chooses
requires a separate quality evaluation.

After the user finishes reviewing, choose one evidence-supported hypothesis,
validate correctness, compare independent complete-call timings and then serving.
The decode compiler-control gap remains in [HANDOFF.md](HANDOFF.md). No sweep,
new kernel, serving rerun or agentic run is started at this checkpoint.
