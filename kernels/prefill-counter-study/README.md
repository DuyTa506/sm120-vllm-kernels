# Minimal scalar L19/M1024 Nsight Compute counter study

Prepared, not profiled. No GPU metric query or capture has run. Existing serving traces contain timings/geometry but no counters that settle DRAM/L2/shared/compute limits. This study proposes no kernel change or performance winner.

One future root-granted command:

```bash
sg docker -c 'bash kernels/prefill-counter-study/run.sh scalar_l19_m1024_counters_1'
```

The launcher holds the shared GPU lock, requires no live GPU compute process and an unused output tag, snapshots helper sources plus portable profiler payload hashes, verifies exact runtime/capture sources, and launches one owned NCU process group in the existing dev container. It does not install anything or modify container/driver/runtime files. The host-installed NCU2025.3.1 target payload (~120MiB) and sections were copied under ignored `tools/`, accessible through the existing/work bind. Its `--version` works in the container with CUDA hidden. No privileged flags or counter permissions are changed.

## Fixed protocol

L19 captured M4096 input prefix1024, current scalar fixed scale preparation, baseline native SHA993783…, M128 workspace, default fast_mathTrue, E256/top8/K2048/I512. Native cache key includes the fixed baseline source SHA. Three complete apply calls warm up outside profiling, followed by an outside-range CUDA event measurement. One NVTX push/pop range `scalar_L19_M1024` contains one full apply+sync. NCU filters only the native gated kernel, not surrounding masking/copies. No CUDA graphs are captured or mutated. Application replay strictly matches the sole selected kernel; each subprocess reloads inputs and reconstructs all persistent queue/workspace state. Unique PID/start-tick directories retain every application pass, while a source-frozen compile cache is shared between passes.

Exactly six requested metrics, pending actual device-supported validation during the granted profiling slot:

- `gpu__time_duration.sum`
- `dram__bytes.sum.per_second`
- `lts__throughput.avg.pct_of_peak_sustained_elapsed`
- `lts__t_sector_hit_rate.pct`
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`

Arguments explicitly set application replay/strict matching, cache-control none, clock-control none, NVTX range filter, kernel-name regex and launch-count1. No `--set full`, profiler metrics query, graph replay mode or source instrumentation is enabled. If a metric is unavailable the run records that failure rather than silently replacing it. Additional shared-memory/stall metrics require later interpretation and a separate reviewed request.

Counter validation imports the saved report without CUDA visibility, requires one native kernel record and all requested metrics with nonempty values, then checks every completed application's output diagnostics and capture/checkpoint identity. Errors compare warmup/profile/fresh outputs for finite relL2<=.02, which is exploratory only. Outside-range timing still runs under profiler injection and is not equivalent to the independent normal microbenchmark. No performance comparison is made from NCU duration.

## Replay and cleanup

Application replay avoids restoring the persistent kernel's queue heads/barrier epochs/scatter state through kernel-replay snapshots. Every pass performs its own complete initialization and warmup. It still changes execution conditions and cannot exactly reproduce40-layer serving cache behavior. Default cache flushing/base-clock locking are explicitly disabled; the first result is a warmed counter diagnostic, not a cold-cache or speed test.

Profiler process PID/PGID/start ticks are recorded. Existing exact-group cleanup is retained; every application replay also records its own PID/start ticks, and cleanup checks these after stopping the profiler group, signals only a still-matching owned PID and fails closed if it survives. Runtime/source/tool hashes are reverified and GPU vacancy is required before exit success. Existing dev container remains running. On failure, full logs and any report/pass artifacts remain.

`ERR_NVGPUCTRPERM` is classified as counter-permission unavailability, unsupported metrics as a tooling/configuration issue, and a missing native kernel as selection failure. None means negative performance. No permission/system changes, tool installation, subsequent shape or autochain is authorized. NCU hardware support/counter access remains untested until a slot is granted.

CPU tests verify fixed argv, six-metric scope, single range/no graph and failure classification. CPU preflight validates13 runtime/helper files plus actual captured operand metadata without CUDA initialization. This is not proof that counters can be collected.
