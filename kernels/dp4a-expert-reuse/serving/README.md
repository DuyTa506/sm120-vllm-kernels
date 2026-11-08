# Native FI direct serving preparation (not deployed)

Default-off `VLLM_SM120_FI_DIRECT_SMALL_M`, registered in actual vLLM compile factors.
CPU OFF→ON→OFF check passed with CUDA uninitialized. `compose.py` creates an
isolated generated overlay; installed runtime and shared scalar sources are unchanged.

Scope: actual checkpoint config snapshot `1c3f884bc99aac2524f6d49bcbac8c88401afd66`,
model_type `qwen3_5_moe`, architecture `Qwen3_5MoeForConditionalGeneration`, text
H2048/I512/E256/top8, BF16 SiLU, TP/DP/EP1. Eligibility is now M1/M3/M4 only; M2 and all
other shapes retain original scalar/CUTLASS behavior. M4 full-call screening passed; live serving gain remains unproven.
The actual repo graph list starts1,2,4,8: MTP2 does not establish serving M3.
Actual ON2 trace observed M3: uniform MTP decode rounds nominal capture sizes up to multiples of query length3. Nominal capture list is not the runtime bucket list.

A persistent resource is keyed by WorkspaceManager execution owner, expert
instance and shape. Existing packed weights, block-scale views and calibrated
alphas are reused. Scalar multiplier planes expand at setup if necessary. Native
handle keeps only required direct scratch/epochs/views, not the large static
workspace or Plan. Sanitizer owns private data/route scratch. The serving wrapper
requires current x/IDs/weights/output on every call and drops warmup pointers.

First creation during graph capture fails. Every required owner must warm both
M1/M4 outside capture before manager locks resources; missing warmup must fail,
not silently allocate. Actual startup owner coverage is not yet verified.

Setup emits source hashes, model identity, live allocation delta, allocator peak
observed (not reset or claimed as isolated operation peak), elapsed setup/JIT time
and sanitizer bytes. No duplicate weight representation is introduced. Timing
must include sanitizer; production runs must retain fixed MTP2 and current patches.
No serving launcher or result is claimed yet. Independent A/B/C live-pointer GPU
checks, M4 timings and actual vLLM trace remain required before promotion.

## Prepared short serving harness (CPU reviewed locally, GPU not granted)

`CACHE_FAMILY=fi_direct_pair1 bash run.sh UNIQUE_TAG off|on` runs exactly one
isolated server with fixed MTP2, CONFIG=next+scalar, fixed imported FI tactics,
max model8192, max sequences8, max batched tokens16384, graph sizes1/2/4/8/16/24/32.
These narrower graph/sequence startup settings apply identically to OFF/ON;
they are a focused serving diagnostic, not the full production configuration.
Original scalar prefill threshold1024 and all other-shape fallbacks are retained.

Tasks mirror repo serving micro workloads: random128→128 decode c1/c4 and
2048→1 prefill c1, deterministic seed6421, exact prompt/output counts required.
Random-token acceptance is workload-specific and not code/QA quality evidence.
Timing precedes a separate Nsight live c1 trace. Trace checks require actual
native direct CUDA graph events correlated to M1/M4 replay NVTX ranges, plus
sanitizer graph nodes when enabled; OFF forbids candidate kernels. No agentic run.

Actual baseline uses V2 `vllm.v1.worker.gpu.model_runner`. Its capture_model at
1053 captures the target, enters draft workspace lane at1105, and locks workspace
at1119 only after non-profile capture. ModelCudaGraphManager runs descriptor
warmup before capture (`gpu/cudagraph_utils.py:459`). Plugin startup verifies the
actual runner/speculator, fixed MTP2, source hashes, compile factors and every
observed owner/layer's M1/M4 resources. A missing shape is an explicit blocker,
not permission to allocate after lock. Live replay counters begin only after
compile_or_warm_up_model returns. Post-workload collective RPC verifies worker
state before shutdown; setup logs contain allocation/JIT costs.

Shared GPU lock, fresh vacancy check, unique owned container, source snapshots,
startup and terminal container state, FI source/tactic checks and fresh GPU
vacancy are mandatory. Failed cleanup changes terminal status; there is no
forced deletion of unrelated containers and no automatic next-profile run.

### First ON startup and repaired configuration hook

`fi_direct_serving_on1` failed during profile-only graph warmup before any serving
workload: `get_current_vllm_config()` is unavailable in forward. Model loading and
torch.compile completed; this is an integration lifecycle failure, not a kernel
numerical result. Source verification passed (candidate and23 parent files),
owned container was removed, and final/fresh GPU compute lists were empty. Exit5
also records the expected missing post-capture FI startup snapshot. Failed source
and logs remain under that run's `experiment_sources_snapshot` and `docker.log`.

Repair captures immutable per-expert model identity in `process_weights_after_loading`,
inside the real Worker.load_model configuration context; forward only requires
that validated metadata. Six CPU tests cover OFF/no getter, missing/wrong/rebound
configuration and immutability. CUDA-hidden inspection of actual Worker.load_model
and the actual checkpoint AutoConfig passed with CUDA uninitialized; compile
factor OFF→ON→OFF still passes. Retry `fi_direct_serving_on2` is prepared but not
run until root grants the GPU. No global permissive config flag or fake context.

### ON2 applicability failure and M3 correction

ON2 reached healthy serving, passed all three task completion guards, then failed
the trace gate:108 actual live graph markers were M3, with no candidate native or
sanitizer kernel. V2 `gpu/cudagraph_utils.py:316` rounds each nominal capture size
up to the uniform decode query length (3 for fixed MTP2). Thus the previous
M1/M4 policy missed c1 decode despite both sizes appearing in startup warmup.
No speedup is inferred from ON2. Sources, FI132 entries and cleanup verified.

The unshipped flag is renamed SMALL_M, with explicit M1/M3/M4 eligibility and M2
fallback. Startup checks require all three shapes per observed owner/layer; live
trace gate now requires native graph kernels correlated specifically with M3.
M3 is added to distinct A/B/C allocation replay checks. Pending focused GPU command:

```bash
sg docker -c 'bash kernels/dp4a-expert-reuse/run_micro.sh dp4a_micro_m3_live1 3 batched live_checks'
```

This runs all18 pointer checks over M1/M3/M4 and seven-provider M3 timing before
another serving retry. CPU14 kernel/helper tests and6 serving tests pass; renamed
compile-factor OFF→ON→OFF passes with CUDA uninitialized. No further GPU run yet.

### ON3 execution proof and future cleanup boundary

ON3 passed startup120handles, all three tasks, actual liveM3 trace4960 native
kernels+4960 sanitizer graph nodes, and explicit post-workload worker RPC. Its
terminal exit5 remains unchanged: shutdown verification lacked a saved
`server_shutdown` checkpoint. Logs show EngineCore receiving SIGTERM, beginning
resource teardown, then the API client's process manager sending another SIGTERM
and reporting the engine manager stopped. Logs do not establish precisely whether
the Worker shutdown hook was never entered or interrupted; "race" is a hypothesis,
not a proven kernel/runtime failure. Container and fresh GPU lists were empty;
all source checks passed. No OFF comparison has yet established a serving gain.

Future launcher cleanup requires `verify_post_workload.py`: exact source bytes,
preload/startup boundaries, startup/post effective132 keys+values and reference
hash, explicit RPC/file identity, same worker, MTP2, liveM3 and all per-owner/layer
resources. Missing post evidence or drift fails. Legacy shutdown verifier is still
run and its report/exit saved, but it is not the measurement-period boundary.
Container/GPU/source cleanup remains mandatory. Kernel, probe and generated
runtime files are byte-identical to ON3; only CPU verifier/launcher cleanup changed.

ON3 active Inductor directories were extracted from actual saved/loaded AOT log
paths: `1caa76...` backbone and `8bed63...` draft. Thirteen selected config identities
are recorded in `active_inductor_configs.json`. The cumulative26-config audit and
2148-file cache manifest are retained as inventory, not evidence those files were
all used. OFF must be compared by its actual active artifact identities/configs.
