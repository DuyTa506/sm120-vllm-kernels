# Handoff — start here

Updated 2026-10-28, 13:30 Asia/Ho_Chi_Minh. The user requested a quick handoff
because progress had become difficult to track. Work is being handed over,
not declared complete. Do not restart the research from scratch.

**Before the next optimization, read [CUDA_PROFILING_WORKFLOW.md](CUDA_PROFILING_WORKFLOW.md).**
It records the NVIDIA APOD / Systems → Compute workflow, correctness tools,
existing trace locations, microbench-to-serving diagnosis and evidence required
for promotion. Added 2026-10-28 without launching another GPU experiment.

**Review checkpoint, 2026-10-28:** the user will read the recorded work before
resuming optimization. Work is paused for that review; do not launch background
GPU jobs. Read [MEMORY_AND_GEMM_NOTES.md](MEMORY_AND_GEMM_NOTES.md) for GDDR/HBM,
cache/shared/register reuse, peak versus useful FLOP/s and the prefill questions
motivated by the new counters. Suggested reading order: this handoff →
[measured progress](PROGRESS.md) → [counter findings](PREFILL_COUNTER_FINDINGS.md)
→ memory/GEMM notes → [profiling workflow](CUDA_PROFILING_WORKFLOW.md).

## The actual outcome

**No round-2 patch has a confirmed serving improvement sufficient for promotion.**

- Native FlashInfer direct is integrated into the actual vLLM MTP-2 **M3**
  decode graph. This is proven by live GPU traces, not just warmup or a Python
  factory hook. Layer-level M3 latency is about **49.26 → 24.63 µs**.
- The short serving pair is mixed and confounded: c1 throughput **+9.59%**,
  c4 **−3.48%**, prefill 2k TTFT **−0.66%**. Four of thirteen active Inductor
  selections differ. Do not present this as a causal speedup.
- Split prefill, workspace tile bands and two operand-load guard variants
  did not beat the current scalar baseline. They are excluded from composition.
- The first Nsight Compute attempt failed with **ERR_NVGPUCTRPERM**. The user
  subsequently supplied a successful attempt2: native scalar L19/M1024 is
  311.68 µs at 1.403 TB/s DRAM. See [counter findings](PREFILL_COUNTER_FINDINGS.md)
  for the report, DRAM-first interpretation and its isolated-kernel limits.
- The current default is still `next`; `scalar` is the existing measured opt-in
  stack, with its previously documented quality/mixed-workload limitations.

The previous agent spent too long refining measurement machinery without
producing another validated serving winner. Keep the next step narrowly tied
to a decision. Avoid another broad fanout of speculative implementations.

## User requirements that remain in force

Target `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` on RTX PRO 6000 Blackwell / SM120,
vLLM 0.32. Prefill has priority; decode also matters. Other models come later.

Test kernels numerically, then complete expert calls, then repository serving
tasks. Compatible patches may stack. **Agentic is only the final selected
composition**, not an evaluation for every individual patch. Publish verified
observations and update the README/chart without turning microbenchmark gains
into serving percentages. The user already authorized local work, benchmarks,
parallel agents and publishing confirmed results; do not repeatedly ask again.

## Where everything is

```text
MAIN=/home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels
EXP=$MAIN/.build/worktrees/qwen-prefill
```

| Location | State |
|---|---|
| MAIN, branch `main` | Published documentation and 53 hash-verified report snapshots at commit `f6f8185` before this handoff commit. |
| EXP, branch `perf/qwen-prefill-workspace` | Base `e535d2e`, with extensive uncommitted experiment code/results. **This is the active implementation workspace.** |
| Remote | `https://github.com/DuyTa506/sm120-vllm-kernels.git`, branch `main`. |

**The public snapshots are not a full implementation archive.** A fresh clone
of main does not contain every current experiment implementation. Preserve this
local worktree; do not reset it, clean it, or merge main over it. No implementation
checkpoint commit was made for the worktree during this handoff.

Local inventories: [worktree status](handoff/worktree-status.txt),
[tracked diff summary](handoff/worktree-diff-stat.txt),
[key-file hashes](handoff/local-file-manifest.json).
These record state, not attribution of every existing edit to this session.

Read [PROGRESS.md](PROGRESS.md) for measured history and
[the report manifest](../results/round2_snapshot/manifest.json) for published evidence.
External repository reviews are already done in
[EXTERNAL_KERNEL_REVIEW.md](EXTERNAL_KERNEL_REVIEW.md) and
[DEEPGEMM_SCHEDULING_REVIEW.md](DEEPGEMM_SCHEDULING_REVIEW.md).

## Environment and execution ownership

- Persistent development container: **`qwen-prefill-dev`**, with EXP mounted at
  `/work`. Preserve this container. Use `sg docker -c '...'` for Docker access.
- Experiment server image: `dg-sm120-env:1`; installed FlashInfer
  `0.7.0.post1`, Torch `2.13.0+cu130`, vLLM `0.32.0`.
- GPU: RTX PRO 6000, about 96 GB, SM120, 188 SMs, 600 W.
- Checkpoint cache: `/opt/intramind/hf-cache`; snapshot
  `1c3f884bc99aac2524f6d49bcbac8c88401afd66`.
- CPU-only Python imports inside Docker must use **`CUDA_VISIBLE_DEVICES=`**.
  A previous source-inspection FlashInfer import unexpectedly created a GPU
  context and contaminated part of a PDL run.
- Standard sandbox exec can fail with `bwrap: loopback ... Operation not
  permitted`; approved escalated exec has worked. This is separate from NCU's
  driver counter permission failure.
- GPU jobs are sequential under the shared lock below. A free GPU must be
  checked before each job. Clean only the exact owned process group/container;
  do not stop unrelated work or the persistent dev container.

```text
/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
```

**At handoff: no active experiment GPU job; fresh GPU process query is empty.**
The last profiler and its application exited; all experimental serving containers
were removed. No automatic benchmark chain is queued.

Two input captures must be preserved:

1. `/work/kernels/moe-nvfp4/data/moe_inputs.pt` inside the container: real
   `hidden[19]`, 8192×2048 BF16, SHA
   `0e161c36a6f478c0205b3a7e8b388296eccafaa2d5999cf4420fcbcd0aa2f4ac`.
   The host worktree file is a zero-byte placeholder; **do not overwrite the
   container's real mounted capture with it**.
2. EXP-relative
   `kernels/moe-task-order/results/task_order_4096_1/measurement/a1/l19_m4096_real.pt`:
   actual input/IDs/weights, SHA
   `9d0b753bc792bf0785087c03e156449a27899697ebb6c4816fe18c82d871abe4`.
   Its adjacent applicability JSON supplies the layer identity.

## Decode: the useful implementation and the unfinished control

All following paths are relative to EXP.

| Path | Purpose / current state |
|---|---|
| `kernels/dp4a-expert-reuse/direct_native.py` | Prepared explicit FI direct launch; no process-global dispatch override. |
| `kernels/dp4a-expert-reuse/direct_sanitize.py` | Padding sanitization and `LiveDirect`; current call supplies input/routes/output pointers. |
| `kernels/dp4a-expert-reuse/live_pointer_checks.py` | 18 checks across M1/M3/M4 with distinct allocations and interleaved sequential graph/eager calls. Passed. |
| `kernels/dp4a-expert-reuse/serving/` | Isolated generated vLLM overlay, launcher, actual-worker probe and strict post-workload verifier. |
| `kernels/dp4a-expert-reuse/serving/inductor_pin/` | **Unfinished compiler control prototype. Read its README before using.** |
| `kernels/fi-static-scale-audit/` | Corrected M4 micro/M8 static comparisons and M8 integration plan. |

Current default-off flag: **`VLLM_SM120_FI_DIRECT_SMALL_M`**, eligible M1/M3/M4
only. M2 and other shapes keep the parent scalar/CUTLASS path. Preserve the
parent global hybrid threshold 1024; lowering it changes other scalar behavior.

Important integration findings:

- **Nominal capture sizes are not effective MTP shapes.** V2 rounds sizes to
  decode query length 3; nominal 1/2 produce actual M3. An initial M1/M4 policy
  warmed 80 handles but executed no candidate in the live request trace.
- Native direct assumes valid expert IDs. Sanitizer maps −1 padding to valid
  ID/zero weight and clears fully invalid hidden rows, including NaNs. Its cost
  is included in the reported full-call timings.
- Bind immutable model identity during `process_weights_after_loading`, under
  the real Worker.load_model config context. Reading current vLLM config in
  forward/warmup failed. Do not introduce a fake context.
- Per-owner/layer/shape handles must warm before workspace lock/capture.
  The successful run has 120 handles = 40 layers × M1/M3/M4. Warmup external
  pointers are released; runtime arguments are not silently reused.
- Direct uses multiplier activation scales. MMA micro/static uses divisor
  scales. A backend-agnostic inversion flag created an invalid all-zero static
  comparator. This was our adapter error; the original repo already documented
  the different contracts. Correct M8 static quantized input matches vLLM bytes.
- Corrected complete-call comparisons favor direct at M4 (~26.7 µs) and static
  at M8 (~34.8 µs). M8 static is **not integrated**; its per-layer scratch would
  add about 0.81 GiB across 40 layers unless safely shared.
- The custom DP4A port passes its checked numerics but generally loses to the
  stronger prepared FI direct path on real routing. Hot synthetic expert reuse
  wins are not serving evidence.

### Exact serving runs

Under `kernels/dp4a-expert-reuse/serving/results/`:

| Run | What it establishes |
|---|---|
| `fi_direct_serving_on1` | Failed on forward-time config lookup. Historical exit 5 retained. |
| `fi_direct_serving_on2` | Three tasks completed; live trace had 108 M3 replays and zero candidate kernels. Applicability failure, exit 1. |
| `fi_direct_serving_on3` | Actual M3 integration: 4960 native + 4960 sanitizer graph kernels. Three tasks and explicit post-workload RPC pass. Historical exit **5** remains because legacy shutdown audit was absent; exact fixed FI table was independently verified at the post-workload boundary. |
| `fi_direct_serving_off1` | Same compute/probe/generated sources, flag off; no candidate kernels, three tasks and strict post-workload checks pass, exit **0**. |
| `fi_direct_pair1_summary.json` | Raw mixed pair; four of thirteen active Inductor configs differ. No promotion. |

Runtime/probe/generated byte parity is in OFF1's `runtime_source_pair_parity.json`.
The later launcher changes only the audit boundary: it requires exact source,
132 FI entries, worker identity, MTP2, stable resources and an explicit post-task
RPC; legacy shutdown audit remains diagnostic. Container/GPU cleanup is still
mandatory. **Never rewrite ON3 exit 5 to 0.**

Focused tasks are random128→128 c1 (8 requests), c4 (16), and 2048→1 c1 (8),
with warmups. MTP2 and FI132 are fixed. Max model length8192/max sequences8 and
a reduced nominal capture list apply to both. This is not the full production
matrix, not the cold-prefill chart, and not a model-quality evaluation.

### What is unfinished in Inductor pinning

`serving/inductor_pin/off1_reference.json` remains **`incomplete_fail_closed`**.
Reference choice is completed OFF1, not whichever table happens to be cached.

- PDL's old semantic normalization collapses distinct zero-fill kernels. The
  new prototype retains generated source filename in its identity.
- All 27 real heuristic constructor checks pass with CUDA uninitialized, in
  both warm and cold checks. Warm constructors all resolve to one config;
  cold constructors have only 14 singletons. Do not mistake warm-cache
  resolution for cold-start pin completeness. Three new negative tests pass.
- 23 of 27 source files have direct AOT/serialized bindings; **four remain
  unresolved**, including two tuning files. Active-directory membership alone
  does not prove a source actually ran. Serialized historical candidates must
  not be mistaken for final selected kernels.
- The pin bootstrap refuses this incomplete reference. **There is no reviewed,
  launch-ready controlled-repeat command.** Do not mark it complete manually.
- Most recent assigned next step: implement a focused actual constructor /
  compiled / first-launch recording pass to resolve those bindings. At handoff
  **the OFF recording launcher has not been implemented**; no valid GPU
  command exists for this next step. All agent files are now frozen.

The main validity problem to resolve is the four changed compiler choices;
sharing the same cache family is not a substitute. Filter actual AOT directories
from model logs, not the cumulative cache inventory. Then repeat on/off with
matched selections before claiming a serving gain or composing this patch.

## Prefill: completed exclusions and successful counter follow-up

| EXP path | Result |
|---|---|
| `kernels/b12x-split-prefill/` | Initial split full call ~17.6% slower at M1024 and ~47.4% slower at M4096. Excluded. |
| `kernels/moe-workspace-bands/` | Smaller tile policy ~1.9% slower at M1024, ~27.1% slower at M2048, with much larger scratch. Native M128 already skips inactive MMA fragments. Excluded. |
| `kernels/moe-inactive-loads/RESULT.md` | 31 load guards make calls ~2.8×/3.4× slower. Stack80→688 bytes; static LDL/STL sites4/5→1484/1099. Excluded. |
| `kernels/moe-inactive-loads-repair/RESULT.md` | Six FC2 guards with explicitly defined operands restore stack80 and LDL/STL4/5, but timing is +0.306%/+1.288% slower. Excluded. |
| `kernels/prefill-counter-study/` | Attempt1 permission failure; user-supplied attempt2 successfully collected counters via a temporary container with SYS_ADMIN. |

Historical first run:
`kernels/prefill-counter-study/results/scalar_l19_m1024_counters_1/`.
`classification.json` says `counter_permission_unavailable`; `run.log` contains
`ERR_NVGPUCTRPERM`; exit1 is preserved. Source71/runtime13/capture verification
and exact profiler/application cleanup pass. No counters were collected.
An outside-range ~919 µs event was measured under profiler injection; **do not
use it as ordinary baseline timing**. No driver, permission or system changes
were made in that first attempt.

**Latest evidence:** `scalar_l19_m1024_counters_2/counters.ncu-rep` was imported
offline after the user collected it. One native kernel, nine application replay
reports, profiler exit0. DRAM 1.402691 TB/s, duration311.68 µs, L2 hit60.39%,
L2 throughput34.25%, tensor/SM20.87%. The [counter findings](PREFILL_COUNTER_FINDINGS.md)
and [published raw evidence](../results/prefill_counter_study_2/manifest.json)
supersede the permission-blocked status for that execution route. This is strong
evidence to target DRAM traffic/throughput at M1024, not a serving-wide diagnosis.
The outer attempt1 audit bundle is not present for attempt2; preserve that limit.

The harness is application replay/strict, one NVTX-filtered baseline kernel,
cache-control none and clock-control none. Its portable NCU payload is a local
copy of the installed tool, under ignored `tools/`; no package installation.
Existing Systems traces provide application context; the new NCU report adds
six metrics for one isolated kernel. Stalls, achieved occupancy and achieved
clocks were not collected. Do not extrapolate to other shapes without evidence.

## Other tracks: do not lose or accidentally promote them

- Humming RN: M1024 layer micro wins ~6.5–10.9% across L0/L19/L39, M4096 loses
  ~33.7%; duplicate layout cost ~16.875 GiB, useful serving bucket reach unproven.
  `kernels/humming-sm120/`; not selected.
- Native task order: hot micro win ~3.9%, actual serving pair no win;
  `kernels/moe-task-order-serving/`. Source-qualified cache keys are essential.
- DeepGEMM FP8 QKVZ: small-shape micro wins, mixed serving; not NVFP4 MoE.
  `kernels/deepgemm-sm120/`; not selected.
- Native swap-AB confirms an existing policy; tiny 64×16 rowwise tile loses.
  PDL focused pair is small/mixed; later c16/c64/NLL possibly contaminated by
  the accidental GPU context. `kernels/pdl-swapab/`; not selected.
- GDN replay pressure/preemption coverage, skinny comparator, MTP scheduling,
  same-checkpoint Marlin and larger composition remain unfinished. Refer to
  existing notes rather than treating CPU preparation as measured success.
- The older final next-vs-scalar agentic pair is already complete. A new agentic
  run belongs only after selecting and validating a new combined stack.

## Resume order

1. Read this file and the [profiling workflow](CUDA_PROFILING_WORKFLOW.md), inspect
   the local inventory, and analyze existing timelines. Preserve experiments.
2. Finish the narrow compiler-control evidence for the real M3 integration;
   record actual constructor/first-launch usage if CPU linkage cannot resolve it.
   Do not launch the incomplete pin table as a controlled benchmark.
3. Run a matched, repeated on/off screen, check quality, then expand to the
   repository task matrix if beneficial. Current raw +9.6% is not confirmed.
4. For prefill, read the successful counter findings and attribute the kernel's
   share of serving prefill time. Choose one traffic/reuse or bandwidth-utilization
   hypothesis; keep full-call timing independent of profiling. Counter access
   works through the user's temporary-container route. Existing negative variants
   stay excluded; no new GPU job is automatically queued.
5. Compose only justified changes, publish the measured configuration/chart,
   and perform the final combined agentic evaluation.

Agent roles retained for context: `new_kernel_directions` owns decode/control;
`pdl_round2` owns prefill/counters; `deepgemm_round2` owns scale/bucket audit and
independent review. They are not autonomous GPU services. Completed agents need
`followup_task` to resume; `send_message` alone does not wake them. This handoff
does not authorize any agent to launch another job in the background.
