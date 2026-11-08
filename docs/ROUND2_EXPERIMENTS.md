# Round-2 experiments: index and outcomes

State as of the 2026-10-28 handoff ([HANDOFF.md](HANDOFF.md)); published 2026-11-08 together with the
experiment code that previously existed only in the local implementation worktree
(branch `perf/qwen-prefill-workspace`).

**Outcome: no round-2 patch has a confirmed serving improvement sufficient for promotion.** The default is
still `next`; `scalar` stays the measured opt-in stack. Several micro-benchmark wins did not reach serving,
and the experiments that did reach it are mixed or confounded (details below). Nothing here is a
production recommendation.

Target throughout: `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast`, vLLM 0.32, one RTX PRO 6000 Blackwell (SM120),
fixed MTP-2. Numbers come from each experiment's README and [PROGRESS.md](PROGRESS.md); micro-benchmark
latencies are hot-layer timings, not request speedups.

## What this repository does and does not contain

| Included | Not included (by choice) |
|---|---|
| Experiment code, harnesses, run scripts, plans and the per-experiment READMEs under `kernels/` | Raw per-run results: ~5,000 JSON/log/CSV files (~214 MB) |
| [Plans and execution record](superpowers/plans/) | Compile caches (torch.compile, Triton, FlashInfer, PDL; ~0.8 GB) and compiled binaries |
| 53 hash-verified report snapshots in [`../results/round2_snapshot/`](../results/round2_snapshot/manifest.json) (its manifest maps each snapshot to its original path) | Nsight traces, generated stack bundles (rebuild with `kernels/stack-round2/prepare.py`), file-hash fingerprint JSONs |

Consequences:

- Links inside the kernel READMEs that point into a `results/` directory refer to raw outputs that are not
  published. The numbers quoted in the README text, in this table and in the snapshots are the evidence available.
- Early paragraphs of some READMEs (for example "no GPU result yet") predate later runs described further
  down the same file. [HANDOFF.md](HANDOFF.md) and the table below are the authoritative status.
- Two real input captures are needed to rerun some numerical checks and are **not** in the repo:
  `kernels/moe-nvfp4/data/moe_inputs.pt` (real `hidden[19]`, 8192x2048 BF16, sha256
  `0e161c36a6f478c0205b3a7e8b388296eccafaa2d5999cf4420fcbcd0aa2f4ac`) and
  `kernels/moe-task-order/results/task_order_4096_1/measurement/a1/l19_m4096_real.pt` (actual input/IDs/weights,
  sha256 `9d0b753bc792bf0785087c03e156449a27899697ebb6c4816fe18c82d871abe4`). Re-capture them from the
  checkpoint before relying on those checks.
- Run scripts take a shared GPU lock whose default path is a stale scratch location (many use
  `${GPU_LOCK:-...}`; others hard-code `LOCK=`). Point it at a writable file first. The DeepGEMM `nv_dev` source
  tree (`kernels/deepgemm-sm120/src`, see `setup.sh`) is a separate clone and is not tracked.

## Prefill

| Experiment | Directory | Idea | Result | Status |
|---|---|---|---|---|
| DeepGEMM rowwise FP8 epilogue | `deepgemm-sm120` | Fused rowwise-scaled FP8 GEMM for QKVZ (K2048/N12288) | 21 actual QKVZ operand cells are 2.70-10.06% faster than the faster of CUTLASS/skinny; nine extra buckets pass; a vLLM trace shows 30 fused calls per M48 replay. Short serving pair is inconclusive: +0.68% decode throughput but +4.43% TPOT, NLL drift, three differing Inductor configs. Large-prefill cells lose | Not selected |
| Native NVFP4 long-first task order | `moe-task-order`, `moe-task-order-diagnostic`, `moe-task-order-serving` | Reorder the native MoE task queue, long tasks first | L19/M4096 full-call ABBA: 434.98/434.30 us -> 417.94/417.65 us (~3.9% lower); weighted BF16 contributions identical, final atomic sums differ by rounding order. Serving pair3 (4096-token prefill): median request latency 69.000 -> 76.754 ms (+11.24%) while summed target-kernel time fell 1.16%; two unchanged Triton producers chose different configs | Not selected; no repeat scheduled |
| Layer-specific MoE dispatch | `moe-layer-dispatch` | Per-layer/shape table choosing the CUTLASS fallback over scalar b12x above the 1024-token threshold | CUTLASS slower in all 12 sampled calibration/held-out cells; numerical gate also failed | No policy; applies only to sampled shapes/routes |
| Split prefill | `b12x-split-prefill` | Split the monolithic native kernel into route/quantization, FC1+SiLU and FC2 stages | Full call ~17.6% slower at M1024, ~47.4% slower at M4096 | Excluded |
| Workspace tile bands | `moe-workspace-bands` | Smaller MoE tile for smaller token bands | ~1.9% slower at M1024, ~27.1% slower at M2048, much larger scratch; native M128 already skips inactive MMA fragments | Excluded |
| Inactive-load guards | `moe-inactive-loads`, `moe-inactive-loads-repair` | Predicate shared-to-register operand copies in entirely inactive MMA warps | 31 guards make calls ~2.8x/3.4x slower (stack 80 -> 688 bytes). Six FC2 guards with defined operands restore the stack but are still +0.306%/+1.288% slower | Excluded |
| Prefill counter study | `prefill-counter-study` | Nsight Compute counters for the native kernel | Attempt 1 failed with `ERR_NVGPUCTRPERM`. Attempt 2 (collected by the user in a privileged container): native scalar L19/M1024 takes 311.68 us at 1.403 TB/s DRAM, L2 hit 60.39%, L2 throughput 34.25%, tensor 20.87% of SM. Points at DRAM traffic for that kernel; one isolated kernel, not a serving-wide diagnosis | Findings in [PREFILL_COUNTER_FINDINGS.md](PREFILL_COUNTER_FINDINGS.md) |
| Humming NVFP4 | `humming-sm120` | Alternative NVFP4 MoE kernel with a private scale-rounding variant ("RN"; the earlier "RU" variant is less accurate) | RN layer micro wins ~6.5-10.9% at M1024 (L0/L19/L39) but loses ~33.7% at M4096; duplicate layout costs ~16.875 GiB. L19/M1024 diagnosis isolates scale rounding: RN matches vLLM FC1 quantized bytes exactly and cuts output relative L2 vs scalar from 17.89% to 2.86%; error vs an FP32 reference is 14.490% (CUTLASS 14.494%, RU 16.329%). The earlier ~8.6% timing gain belongs to RU, not RN | Not selected; no model speedup claimed |

## Decode

| Experiment | Directory | Idea | Result | Status |
|---|---|---|---|---|
| Native FlashInfer direct for small M | `dp4a-expert-reuse`, `fi-static-scale-audit` | Use the prepared native direct MoE launch for M1/M3/M4 (`VLLM_SM120_FI_DIRECT_SMALL_M`, default off) | Layer-level M3 49.26 -> 24.63 us. Proven to execute inside the live MTP-2 M3 graph (120 handles = 40 layers x M1/M3/M4; 4,960 native + 4,960 sanitizer graph kernels). Short serving pair: c1 +9.59%, c4 -3.48%, 2k TTFT -0.66%, but 4 of 13 active Inductor selections differ, so it is not a causal speedup. Corrected L19 bucket comparison: M4 direct 26.67 us vs MMA 28.70 vs CUTLASS 61.2; M8 static 34.8 us vs direct 43.0 vs CUTLASS 63.5 | Not promoted. Inductor-pinning control unfinished (`serving/inductor_pin/`, fail-closed, 23 of 27 source bindings resolved) |
| DP4A expert reuse | `dp4a-expert-reuse` | Deduplicate experts across tokens (custom port) | Numerics exact on M1 and M8 checks (graph/eager, reuse on/off, padding); generally loses to prepared FI direct on real routing; hot synthetic reuse wins are not serving evidence | Superseded by the direct path |
| PDL / swap-AB | `pdl-swapab` | PDL overlap and swap-AB tiles (ideas from DeepGEMM) | PDL artifact/config proof passes, NLL shift unresolved, focused pair small and mixed (later runs possibly contaminated by an accidental GPU context). Native swap matrices: all 16 off/on outputs identical, QKVZ-shaped N12288/K2048 latency 2.3-4.1% lower, which confirms an existing policy. Tiny rowwise FP8 swap: L0/M3 AUTO 17.77-17.81 us vs skinny 17.00-17.04 us; a 64x16 tile takes 19.62 us (~11.2% slower than its AUTO control) | Not selected |
| GDN state replay | `gdn-replay/round2` | Replay only accepted tokens into one state slot | Current-stack shadow `replay_r2_shadow3` passes: 869,070 checked state rows incl. 630 intermediate-boundary rows, zero state/output/boundary errors; 32 boundary requests produced 2,304 tokens each; stress 91 completed / 37 aborted without failures. Pressure run `replay_r2_preempt1` exited 1 because no preemption occurred (scheduler admitted ~4 requests and queued the rest). Paired serving timing pending; historical gains (c=64 ITL 28.6 -> 26.2 ms) are not new-stack results | Unfinished |
| Skinny BF16/FP8 on current stack | `decode-roofline/round2` | Re-validate the historical skinny GEMMs on scalar + next | Compile-key fix (flags registered in `compile_factors`) checked on CPU: five distinct hashes for off/BF16/FP8/both/MAX_M32. No warm-cache or serving result was measured | CPU preparation only |
| MTP depth scheduling | `mtp-scheduling` | Fixed MTP-2 vs the installed schedule `[[1,8,2],[9,64,1]]` | Implemented and CPU-checked; not GPU-tested | No gain claimed |
| Same-checkpoint Marlin control | `marlin-control` | W4A4 vs W4A16 on the same NVFP4 checkpoint | CPU preparation and eligibility audit only | Not run |

## Composition

`stack-round2` builds isolated overlays (parent, task_order, dense, combined) from the pieces above. The
current revision is `ready2`; earlier bundles are provenance only. The stack itself has not run on GPU, and
the [execution record](superpowers/plans/2026-10-28-sm120-round2-execution.md) lists the failed runs and the
queue at handoff. The final agentic evaluation is reserved for a selected composition and has not been run
for any round-2 candidate.

## Where to start

1. [HANDOFF.md](HANDOFF.md), then [PROGRESS.md](PROGRESS.md) for measured history.
2. [CUDA_PROFILING_WORKFLOW.md](CUDA_PROFILING_WORKFLOW.md) and [PREFILL_COUNTER_FINDINGS.md](PREFILL_COUNTER_FINDINGS.md)
   before choosing the next prefill hypothesis (DRAM traffic at M1024).
3. For decode: finish the Inductor-pinning control for the native direct path, then repeat the on/off pair with
   matched compiler selections before claiming a gain.
4. Candidate inventory and external reviews: [`superpowers/plans/`](superpowers/plans/),
   [EXTERNAL_KERNEL_REVIEW.md](EXTERNAL_KERNEL_REVIEW.md), [DEEPGEMM_SCHEDULING_REVIEW.md](DEEPGEMM_SCHEDULING_REVIEW.md).
