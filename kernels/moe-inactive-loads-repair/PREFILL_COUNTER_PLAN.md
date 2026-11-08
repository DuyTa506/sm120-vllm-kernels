# Measure the current scalar prefill bottleneck before another kernel patch

Read-only inventory: Nsight Compute2025.3.1.0 at `/usr/local/cuda/bin/ncu`; Nsight Systems2025.3.2.474 at `/usr/local/cuda/bin/nsys`. Version/help and installed documentation/section files were read; no profiler hardware query, capture, install or GPU work occurred. No `.ncu-rep` or saved NCU counter CSV/log was found in the worktree/main kernel result trees searched. Existing traces do not settle the hardware bottleneck.

## Existing actual serving evidence

`kernels/moe-task-order-serving/results/task_order_serving_a3/traces/prefill4096.sqlite` contains40 actual baseline gated native kernel nodes, median431.756µs across all layers. Recorded launch geometry is grid(1,1,188), block(288,1,1), registersPerThread168 and dynamicSharedMemory101376 bytes. This is useful activation/duration/resource evidence, not a measured DRAM or shared-memory saturation result. The database has no metric/counter/sampling tables. Its staticSharedMemory field is0, whereas extracted ELF resources report1024 static bytes; preserve these as distinct tool views rather than substituting one for the other. Aggregated durations across the40 layers are not L19-specific hardware counters or a critical-path attribution.

The kernel is persistent and internally fuses routing/packing, queue publication, FC1/quantization, FC2 and atomic scatter. A single kernel average can hide phases with different bottlenecks. Large allocation or weight byte counts alone do not establish DRAM limitation.

## Minimal first measurement

Use the existing frozen baseline scalar preparation and captured L19/M4096 operands. Profile M1024 prefix and M4096 separately, M128 workspace, native source SHA993783… and serving-default fast_mathTrue. The capture contains only4096 rows; do not duplicate it and label the result realM8192. Add M8192 only after obtaining an actual8192-token capture or locating a verified existing one.

Prepare one small baseline-only script per requested shape: load exact weights/scales, warm complete `apply` calls outside the profiling range, then enter one named NVTX range and issue one complete call. Capture/verify baseline graph execution separately to bind the same native artifact and geometry; first counter pass can use the eager native launch to avoid CUDA graph replay ambiguity. Preserve full-call unprofiled timing and output checks before/after. Profile only the matching native kernel in that NVTX range; do not include compilation/warmup or every MoE layer.

Suggested **future** command shape, not executed and requiring a new reviewed runner:

```text
ncu --replay-mode application --app-replay-mode strict \
    --cache-control none --clock-control none \
    --nvtx --nvtx-include 'scalar_L19_M1024/' \
    --kernel-name 'regex:.*gatedMoEGatedDynamicKernel.*' --launch-count 1 \
    --metrics <small verified SM120 metric group> \
    --export <unique report> python3 <baseline-only runner> --tokens 1024
```

Start with duration, DRAM bytes/rate, L2 throughput/hit rate, aggregate SM throughput and tensor-pipe activity. Installed section definitions supply these candidate names: `gpu__time_duration.sum`, `dram__bytes.sum.per_second`, `lts__throughput.avg.pct_of_peak_sustained_elapsed`, `lts__t_sector_hit_rate.pct`, `sm__throughput.avg.pct_of_peak_sustained_elapsed`, and `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed` (with an architecture-dependent v2 fallback). Resolve actual SM120 availability only under a later profiling slot; no unsupported-metric substitution or broad `--set full` first. Split groups if the tool reports excessive passes. Use the installed architecture-filtered section definitions instead of assuming every named metric exists on SM120.

If initial results do not show a saturated pipe, add a second focused group: eligible warps, issue-active, shared-memory wavefronts/bank conflicts, and short/long-scoreboard/barrier/membar stall ratios. Installed metric names include `smsp__warps_eligible.avg.per_cycle_active`, `smsp__issue_active.avg.per_cycle_active`, and `smsp__average_warps_issue_stalled_{short_scoreboard,long_scoreboard,barrier,membar}_per_issue_active.ratio`. Shared-memory metrics should be selected from the device-supported memory analysis table. Interpret stalls only together with poor issue utilization; a large stall ratio alone is not lost throughput.

Decision distinctions:

- DRAM rate near the attainable sustained rate plus substantial DRAM bytes and long-scoreboard pressure supports a global-memory limit. A high generic memory percentage alone is insufficient.
- High L2 throughput with high hit rate and low DRAM traffic points toward cache-side traffic, not an HBM/DRAM bound.
- Shared-memory wavefront/bank-conflict pressure and short-scoreboard stalls with limited DRAM/tensor activity can support a shared-memory load/layout question.
- High tensor-pipe occupancy with low memory pressure supports compute work as a target; aggregate SM throughput also includes non-tensor instructions.
- Low eligible/issue utilization with barriers or scoreboard stalls requires phase/PC attribution before blaming producer synchronization. Grid-barrier polling and ready-queue spins use load/atomic loops and can appear as scoreboard/instruction activity, not just the hardware `barrier` stall category. If needed, a later PC/warp-state sample should map addresses to the exact saved SASS and producer/consumer regions. Do not add phase instrumentation or assume an aggregate counter identifies FC1 versus FC2 on the first pass.

## Replay and measurement limits

Installed NVIDIA ProfilingGuide explains that kernel replay saves accessible memory and restores written allocations between passes; persistent queue heads, counts, epochs, packed buffers and scatter outputs must all be restored consistently. This also adds traffic/state effects. Application replay is preferable for the first screen: each pass reconstructs the baseline process and runs its own initialization/warmup, avoiding dependence on replay restoration of persistent workspaces. Require strict kernel matching and identical input/source hashes, but do not demand bitwise outputs from the nondeterministic atomic reduction. Use fresh per-process output directories so profiler re-launches do not fail an existing-output guard. A frozen native compile cache may be reused to avoid repeated compilation; it must not select different code between passes.

Default NCU cache-control is`all` and clock-control is`base`. Override both to`none` explicitly for a warmed-workload diagnostic, record actual clocks through normal permitted telemetry during the future slot, and keep an unprofiled reference. Application replay can preserve the preceding warmup's cache state; it still changes process execution and cannot reproduce the whole serving inter-layer cache state perfectly. Do not compare NCU replay duration directly with normal serving latency as a speed claim. A deliberate cold-cache study is a separate named protocol, not an unnoticed profiler default.

Never split the persistent kernel into independently profiled CTA pieces: its grid-wide producer barriers require resident cooperative behavior across all188 CTAs. Hardware-counter collection must retain the entire original launch geometry. Avoid first-pass SASS instrumentation, broad metric sets and PC sampling that may change overhead/resources. If counter permissions fail, preserve `ERR_NVGPUCTRPERM` and report the limitation; no tool installation or permission change is authorized by this study.

Primary local references: `/opt/nvidia/nsight-compute/2025.3.1/docs/ProfilingGuide/index.html` sections Kernel Replay, Application Replay, Range Replay and Cache Control; `sections/SpeedOfLight.section`, `MemoryWorkloadAnalysis.section`, `ComputeWorkloadAnalysis.section`, `SchedulerStatistics.section`, and `WarpStateStatistics.section`. No existing counters presently justify another load/tile mutation.
