# First counter attempt: permission unavailable

`scalar_l19_m1024_counters_1` completed with profiler exit1 and classification `counter_permission_unavailable`. Nsight Compute reported `ERR_NVGPUCTRPERM`: the process could not access NVIDIA GPU Performance Counters on device0. No requested counter report was collected, so this run does not distinguish DRAM/L2/shared-memory/compute/producer-barrier limits and is not a negative performance result.

The exact baseline app ran through its sole NVTX range and completed exploratory output checks: before/after relL2=.00427320 and profiled/fresh relL2=.00404801, finite. Its recorded outside-range event time919.2µs was measured under profiler injection and must not be substituted for the normal full-call314µs baseline or used as a slowdown claim. No production numerical equivalence claim follows.

The profiler also warned that stock section deployment under its default Documents location failed and that caches/clocks were uncontrolled. The explicit copied section path and fixed metric list were passed; cache-control none and clock-control none were deliberate. The terminal cause was counter permission, not an unsupported metric or a changed workload. Device support for the complete requested metric group remains unverified.

Frozen71 source/tool hashes and13 runtime/helper hashes plus capture identity remained unchanged. Profiler PID/PGID37835 exited, its process group was empty, recorded application PID37843 was absent, and final plus fresh GPU queries were empty. GPU lease was released. Full logs, command arguments, application reports and cleanup records are preserved in the result directory.

No retry, additional shape, privilege/capability change, package installation or driver/system configuration change was performed. Existing timing/trace/source evidence remains the available basis; new hardware-counter conclusions require a separately authorized way to access counters.
