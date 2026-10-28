# CUDA profiling workflow for the next researcher

Recorded 2026-10-28. Read [HANDOFF.md](HANDOFF.md) for experiment state and
[PROGRESS.md](PROGRESS.md) for measured results. This document adds a method;
it does not announce a new speedup or restart any GPU job.

The foundation is NVIDIA's **Assess, Parallelize, Optimize, Deploy (APOD)** cycle:
identify important application costs, change the implementation, validate, and
repeat. NVIDIA's kernel triage guide specifically recommends Systems before
Compute when kernel priority is unclear, one targeted change at a time, and
judging absolute duration rather than utilization percentages alone.
[CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#assess-parallelize-optimize-deploy),
[Compute Triage](https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html).

The workload controls and promotion rules below are this repository's application
of that guidance, not a claim that NVIDIA prescribes our exact benchmark matrix.

## 1. Establish the question before changing a kernel

Follow this chain of evidence:

```text
Unprofiled serving baseline → application timeline → costly operation
    → kernel diagnosis → one change + correctness → complete-call microbench
    → actual vLLM dispatch + GPU step → unprofiled serving → selected composition
```

Write one falsifiable hypothesis, the evidence behind it, and the measurement
that could reject it. For example: “FC2 operand loading limits the measured
prefill step; reducing that traffic should reduce complete-call and step time.”
The presence of padding or an attractive instruction sequence is not evidence
that it limits execution. If three variants fail without explaining the trace,
return to diagnosis rather than adding more variants.

Separate the workloads before interpreting results:

| Workload | Record and compare | Common misleading shortcut |
|---|---|---|
| Cold prefill | Actual uncached input tokens, chunk sizes, GPU prefill span, TTFT, prompt throughput | Calling every 2048-token request a cold M2048 GEMM |
| Decode | Actual query-token M, live/padded rows, context lengths, c1 and higher concurrency, TPOT/ITL, output throughput | Treating request count or nominal graph size as M |
| Mixed serving | Arrival schedule, queue time, batch composition, TTFT/TPOT distributions, total completed tokens | Inferring mixed throughput from an isolated layer |
| MTP | Target/draft work, proposed/accepted tokens, number of forward steps per output token | Equating a faster forward with faster generation |

## 2. Freeze a reproducible unprofiled baseline

Save the model/checkpoint, patch and runtime source identities, library/tool
versions, effective backend/tactics and **actually used** compiler selections.
Include quantization/scaling contract, graph mode, MTP settings, batch limits,
prefix-cache policy and actual hits, KV capacity, prompt/output lengths, seeds,
concurrency or arrival rate, warmup, and timing boundaries.

Use the existing launchers and shared GPU lock from the handoff. Record other
GPU processes, clocks, power/temperature and memory use; do not silently change
clock or power policy. Contended measurements may screen ideas but need their
noise and limitation recorded. Two favorable passes alone do not remove drift.
CPU-only inspection must use `CUDA_VISIBLE_DEVICES=` to avoid accidental contexts.

Measure startup/JIT/packing separately from steady state. If preparation or
duplicate weights change available KV cache, include that serving consequence.
Use repeated, interleaved A/B or ABBA trials with matched warmup/cache policy.
Report all trial summaries and dispersion, preferably a paired confidence
interval across independent trials. Thousands of launches within one warmed
process are not thousands of independent serving trials. Choose an improvement
threshold relative to observed noise before deciding which result to keep.

**Keep timing runs separate from profiler runs.** A profiler trace explains
behavior; unprofiled measurements establish the performance claim.

## 3. Tool map

| Tool | Question it answers | Scope and current local state |
|---|---|---|
| CUDA events + host monotonic timing | How long does GPU work / the complete request take? | Events must enclose the intended stream dependencies; synchronize the completion event before reading. Host timing of an async launch alone is not GPU time. Use the existing harness. |
| Nsight Systems (`nsys`) + NVTX | Which CPU/GPU work, gaps and dependencies dominate the application? Does the candidate run? | Host 2025.3.2; live ON3/OFF1 reports already exist. Start here. |
| Nsight Compute (`ncu`) | Which resource or latency mechanism limits a selected kernel? | Host 2025.3.1; local container payload exists. First collection failed with `ERR_NVGPUCTRPERM`; no counters. |
| PyTorch Profiler / vLLM profiling hooks | Which framework operator, shape or call stack produced the work? | Optional focused diagnostic when CUDA names are insufficient; richer instrumentation adds overhead. Not a replacement for normal timing. |
| Compute Sanitizer | Is there an invalid access, uninitialized read, shared-memory hazard or synchronization error? | Host executable 2025.3.1 verified by `--version`; this handoff does not claim a sanitizer run. |
| `cuobjdump`, `nvdisasm`, compiler resource reports | Did the compiler emit the intended MMA, loads, barriers and resource usage? | Available on host; offline disassembly was used for the load-guard experiment. Static instruction sites are not executed instruction counts. |
| `nvidia-smi`, serving metrics and request logs | Was the GPU contended, the queue growing, or token/step accounting different? | Context for the measurement, not proof of a kernel's bandwidth or compute limit. |

NVIDIA documents timeline tracing in the [Systems guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
and binary inspection in [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/).
vLLM documents its [PyTorch and Nsight profiling integration](https://docs.vllm.ai/en/stable/contributing/profiling/).
Online documentation may be newer than our installed versions; check local help
and runtime source before copying new flags or APIs.

## 4. Find the application bottleneck with Nsight Systems

Capture a short representative steady-state window after warmup, with the real
worker processes included. Keep startup/JIT out unless startup is the question.
vLLM recommends `VLLM_WORKER_MULTIPROC_METHOD=spawn`,
`--trace-fork-before-exec=true` and `--cuda-graph-trace=node`; node tracing exposes
individual graph kernels but adds overhead. Preserve the existing launcher's
capture boundaries rather than profiling the entire model load.
[vLLM profiling](https://docs.vllm.ai/en/stable/contributing/profiling/).

Use NVTX or an aligned metadata sidecar to identify request/step, phase, layer,
actual M, valid rows, graph bucket, backend and target versus draft. Suggested
operation boundaries are routing/top-k, dispatch/packing/quantization, FC1,
activation/requantization, FC2, combine and shared expert. Also distinguish
attention/GDN, input preparation, sampling, scheduling and output handling.
These are proposed coverage points; the current traces do not contain all of them.
Collect expert-count/skew summaries outside timed regions or with explicitly
measured instrumentation overhead; avoid `.item()`/device-to-host synchronization
on every decode step just to produce a log.

Inspect kernel applicability first, then launch count, per-operation duration,
GPU step envelope and dependencies, idle gaps, transfers and synchronization.
Correlate host ranges through CUDA API correlation IDs and graph-node identity
to GPU execution; when querying SQLite, include process/context identity where
available so IDs from separate workers cannot be accidentally joined.

Interpretation rules for our traces:

- CPU NVTX duration measures the host region. Async graph execution can outlive it.
- A GPU envelope is first relevant GPU start to last relevant GPU end; it is a
  useful step measure, not by itself a dependency-critical-path reconstruction.
- Summed kernel durations can exceed elapsed time because streams overlap.
  Do not add them or component microbenchmarks to manufacture request latency.
- Compare matched phases/shapes and normalize by steps and accepted/output tokens.
  ON and OFF may execute different numbers of steps; whole-trace totals mislead.
- Cache-file presence and startup warmup do not prove live kernel execution.
  Check candidate symbols/nodes in requests, fallback coverage, and compiler drift.

Systems provides kernel, CUDA API and NVTX summaries plus SQLite export for
custom attribution. Those summaries locate candidates; the timeline establishes
their relationship. [Systems analysis guide](https://docs.nvidia.com/nsight-systems/AnalysisGuide/index.html).

## 5. Diagnose the selected kernel with Nsight Compute

Profile one known kernel/shape/input first, after Systems shows it matters.
Use a release build, with line information where supported, and preserve the
binary identity. Start with a small set of relevant sections; query supported
metrics for the installed tool/SM120 instead of copying another architecture's
metric names. Record replay mode, cache and clock controls. Persistent queues
and barriers need correct reinitialization across replay passes; application
replay must reproduce the selected work. Profiler timing is not the independent
benchmark. [NCU Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

| Suspected limiter | Evidence to seek before choosing a change |
|---|---|
| Too little parallel work / tail effects | Grid versus available SMs, waves, uneven expert work; compare actual shapes |
| DRAM or cache throughput | Bytes/time and throughput breakdown at DRAM, L2 and L1; distinguish traffic reduction from a higher hit rate |
| Memory latency | No saturated tier, limited eligible warps/issue, load dependencies; distinguish insufficient outstanding work from a bandwidth ceiling |
| Compute / instruction issue | Dominant pipeline, instruction mix and tensor activity, useful versus padded work |
| Registers / shared memory / synchronization | Resource limits, local traffic, bank conflicts, barrier dependencies and source locations |

Use multiple indicators; low occupancy or a large stall percentage alone is not
a diagnosis. Roofline estimates must state useful FLOPs, bytes, scale/metadata
traffic and reuse assumptions. Model-size divided by advertised bandwidth is not
proof that decode has reached its ceiling. Route skew and weight reuse matter,
but so can launch latency, other layers and MTP. Reordering dispatch must preserve
the model's selected experts and weights; changing routing is a quality change.
[NVIDIA Compute Triage](https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html).

**Current blocker:** the prepared prefill study failed because GPU performance
counters are unavailable to the profiling process. This is not a negative kernel
result. Do not rerun unchanged or infer counter values. Counter access needs an
explicit administrator/container-policy resolution before retry; this note
does not change those permissions. Systems CUDA/NVTX analysis remains usable.
[NVIDIA's error explanation](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-nsightcompute).

## 6. Check correctness, then measure one change

Define the mathematical/quantization contract first. Compare against an explicit
reference and the current backend with justified tolerances, not merely one
another. Cover real inputs, empty/skewed experts, padding, partial tiles, supported
shapes, repeated calls, graph/eager execution and changing live pointers. Preserve
NVFP4 scale granularity, multiplier/divisor conventions, global scaling and
rounding; a faster mismatched comparator is invalid.

For a small kernel reproducer, use Compute Sanitizer's `memcheck` for access errors,
`racecheck` for shared-memory hazards, `initcheck` for uninitialized device global
reads and `synccheck` for synchronization misuse. Run relevant checks separately;
these instrumented runs are correctness diagnostics, not timing trials. A clean
run is limited to tested executions and does not prove numerical or model quality.
[Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html).

Keep one independent implementation change per ablation. Microbench the entire
affected call, including required sanitization, routing/packing, quantization,
epilogues and reduction; state every exclusion. Use captured inputs/routes and
representative layer/cache conditions, and distinguish eager from graph replay.
Compare emitted resource use/SASS when a source simplification regresses.

## 7. Explain microbench versus serving before promotion

| Observation | Next discriminating check |
|---|---|
| Micro win, candidate absent from live trace | Actual eligibility/M, graph rounding, dispatch and cached implementation identity |
| Candidate slower only in serving | Match shapes/routes/layout; inspect cache reuse, scratch, competing streams and preparation cost |
| Kernel faster, complete operation unchanged | Include packing, quantization, sanitizer, reduction and allocation costs |
| Operation faster, GPU step unchanged | Inspect other operators, overlap and dependency path; estimate eligible coverage |
| GPU step faster, request throughput unchanged | Compare queue/CPU work and target/draft/accepted-token accounting; do not assume the CPU is guilty |
| c1 improves, c4 regresses | Compare M buckets, fallback path, batch composition, MTP acceptance and queueing |
| Compiler selections or numerical outputs differ | Separate configuration-level observations from a controlled patch comparison |

Estimate the available headroom. If a component accounts for fraction `f` of a
fixed serial elapsed path and becomes `s` times faster, the ideal speedup is
`1 / ((1 - f) + f / s)`. A 2× improvement in a 10% component yields only 1.053×.
This is a planning estimate; overlapping work and changing batching/acceptance
require the actual timeline and output-token accounting.
[Amdahl's law in CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#strong-scaling-and-amdahl-s-law).

Promote only after repeatable unprofiled serving gains, numerical/quality checks,
actual dispatch proof and acceptable regressions/memory cost. A mixed result can
justify a measured shape-specific opt-in, but not an unconditional default.
Expand a successful focused screen to the repository task matrix. Compose
compatible accepted changes, remeasure the combined stack, and run **agentic only
for that final selected composition**. Speedups from separate patches do not add.

## 8. Existing artifacts: start here without another GPU run

Paths below are relative to
`EXP=.build/worktrees/qwen-prefill` from the repository root. They are local
artifacts, not all available in a fresh clone; preserve the dirty worktree.
Some experiment READMEs retain earlier “not run” entries: use the current handoff
and terminal result files for status.

| Artifact under EXP | What it establishes / next use |
|---|---|
| `kernels/dp4a-expert-reuse/serving/results/fi_direct_serving_on3/traces/live.nsys-rep` and `live.sqlite` | Candidate executes in live M3: 4960 direct + 4960 padding-sanitizer graph kernels. The padding sanitizer here is our kernel, **not** NVIDIA Compute Sanitizer. |
| `kernels/dp4a-expert-reuse/serving/results/fi_direct_serving_off1/traces/live.nsys-rep` and `live.sqlite` | Baseline timeline for operation/step attribution. Pair is confounded by 4/13 active Inductor selection differences. |
| `kernels/dp4a-expert-reuse/serving/inductor_pin/` | Unfinished control prototype; reference is `incomplete_fail_closed`, not ready for a controlled GPU run. |
| `kernels/prefill-counter-study/results/scalar_l19_m1024_counters_1/` | `classification.json`/`run.log`: counter permission failure, exit1, no counters. Outside-range ~919 µs timing under injection is not a normal baseline. |
| `kernels/prefill-counter-study/` | Existing focused application-replay harness, NVTX filter, source/capture checks and cleanup; consult it after counter access is resolved. |

Current conclusions remain those in the [handoff](HANDOFF.md): M3 layer micro
49.26→24.63 µs and live integration are established; the short c1/c4 serving pair
is mixed and confounded, not a new promoted winner. The trace applicability check
is complete; a controlled explanation of the whole serving delta is not.
ON3's historical exit5 remains recorded; post-workload verification does not
rewrite its exit status.

These commands only analyze an existing SQLite trace (no GPU workload). Run from
the repository root; substitute OFF1 to inspect the other side. Report names were
checked against the installed Systems 2025.3.2 help.

```bash
NSYS=/opt/nvidia/nsight-systems/2025.3.2/bin/nsys
TRACE_BASE=.build/worktrees/qwen-prefill/kernels/dp4a-expert-reuse/serving/results
TRACE_SQLITE="$TRACE_BASE/fi_direct_serving_on3/traces/live.sqlite"
"$NSYS" stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum "$TRACE_SQLITE"
"$NSYS" stats --report cuda_kern_exec_sum,cuda_gpu_kern_gb_sum "$TRACE_SQLITE"
```

Open `.nsys-rep` in the Systems GUI for timeline/dependency inspection. For custom
SQLite queries use read-only mode, inspect the schema, and preserve raw reports.
Do not average unlike phases or count nested NVTX regions twice.

The official vLLM [`gputrc2graph.py`](https://github.com/vllm-project/vllm/blob/main/tools/profiler/nsys_profile_tools/README.md)
can produce CSV/HTML from Systems reports. It accepts an unprofiled elapsed time;
using trace elapsed instead can inflate apparent non-GPU time. Audit its model
classifier for Qwen's GDN/MoE/draft kernels before trusting categories; do not
force every CUTLASS symbol into “MoE.” This tool is a follow-up option, not a
completed analysis in this handoff.

## 9. Minimum record for each next experiment

```text
Question / falsifiable hypothesis:
Trace evidence + expected headroom:
Baseline identity / one changed variable / actual eligible workload:
Correctness reference, tolerances and tested cases:
Timing scope, warmup/cache policy, repetitions and GPU conditions:
Micro / complete-call / GPU-step / unprofiled serving results + uncertainty:
Actual dispatch, compiler controls, MTP token accounting and quality limits:
Reproduction command + source/config identities + raw artifact paths:
Decision: retain / conditional / reject / unresolved; next discriminating check:
```

The next agent should first analyze existing timelines and resolve the narrow
M3 compiler-control gap. For prefill, use Systems to rank costs while the NCU
permission issue remains unresolved. Choose the next kernel experiment from that
evidence. No background GPU run or permission change is started by this handoff.
