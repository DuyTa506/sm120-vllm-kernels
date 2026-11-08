# Native NVFP4 long-task-first queue experiment

Status: native M4096 ABBA screen completed with about3.9% lower full-call latency, but cross-variant output drift exceeds baseline repeat variation. This is a performance lead, not a numerically approved or serving-ready patch. This is a new task-order experiment on native b12x, based on the DeepGEMM agent's read-only `NATIVE_NVFP4_NEXT.md` audit. Existing GDN/skinny preparations remain untouched.

The gated dynamic kernel already splits a prefix of physical M tiles for M257..4096 to approach roughly four resident waves (a slightly different target above2048). It currently publishes those short tasks first, followed by unsplit longer tasks. This experiment changes only descriptor slot order:

- Unsplit tile t≥S starts at t−S.
- Split tile t<S starts at (T−S)+tG, group g goes to start+g.

T is the existing device-computed total physical tile count, S is the existing split count and G=ceil(gate_tiles/2). Descriptor values, tile/expert indices, valid rows, split boundaries, task count, publication fences, queue consumers, native per16 E4M3 scales, activation permutation, MMA and local epilogues are unchanged. No learned routing choice changes and no host decision/read is added to forward. S=0 or S=T is the identity permutation, so those workloads are applicability controls rather than expected wins.

`prepare.py` saves baseline/candidate hashes and asserts only two method ASTs differ: `publish_variable_deferred_tasks` and its producer callsite. A distinct candidate class name avoids conflating compiled class identities. Default `micro.py --variant baseline` uses the unmodified installed class after hash checking; explicit `long_first` loads the isolated candidate and redirects only the process-local factory before any compilation. Installed files are never edited. Each profile must run in a fresh process; no in-process baseline/candidate swap. A private per-run compile-cache namespace should be used by the root launcher before GPU execution; disk-cache isolation/runtime class verification must be reviewed before granting the first smoke.

CPU tests cover total tiles0..40, every split count and gate-tile counts1..9, including odd last groups: no duplicate slots/holes, equal task counts and identical packed descriptor multisets. Three tests pass. The test models variable tasks only; below1024 current vLLM scalar backend still uses CUTLASS and is not redirected.

## First full-call screen

`micro.py` loads real checkpoint weights and captured hidden states, recomputes actual learned routes, preserves calibrated scalar input scales and calls the complete existing scalar backend (quantization/route packing, publication, FC1/Q1/FC2 and finalization included). Initial L19/M1024, then M2048 and layers0/39; no GPU request has been issued yet. Real routing is primary. Uniform and eight-hot controls keep valid exact top8 choices but are synthetic, used to expose whether the queue has both split and unsplit tasks.

After each completed call, outside timing/capture, the harness reads task metadata, derives S from the actual published count and verifies every packed GPU descriptor against the CPU model for the selected order. It records T/S/G, task count and whether the permutation is nontrivial; results with S0 or ST do not support the scheduling hypothesis. CUDA graph timing includes the full call. Constructor tracing asserts the selected native class; no alternative backend is silently accepted.

Suggested first command after a root-managed locked launcher and grant:

```bash
python3 /work/kernels/moe-task-order/micro.py --variant baseline --layers 19 --tokens 1024 --patterns real --output /work/kernels/moe-task-order/results/a1
```

Then candidate b1, candidate b2, baseline a2 under one controlled sequential lease, separate fresh processes and output directories. Do not launch from this README without the GPU scheduler grant. Add skew/uniform controls and larger captured layers only if the initial queue is nontrivial and the candidate works.

## Numerical constraints

Final scatter uses BF16 atomic additions. Reordering tasks can change reduction order; this cannot claim bitwise identical global output. `micro.py` saves eight repeated same-input outputs for each variant. `compare.py --baseline A1 A2 --candidate B1 B2` verifies identical input/route bytes and descriptor multisets, then reports baseline–baseline, candidate–candidate and baseline–candidate relative-L2 distributions with paired timings. It does not automatically relax accuracy gates or promote the patch. A shift beyond baseline self-variation requires investigation; equal-looking distributions alone do not establish model quality. Any surviving speed candidate still needs exact-operand reference checks and existing vLLM NLL/quality tasks before serving or combined agentic measurement.

Potential failure: long-first may worsen locality, make early CTAs work longer or reduce beneficial overlap. A negative outcome is specific to observed task mixtures and token/routing shapes; it is not evidence that all native NVFP4 optimization is exhausted.

GPU-ready first screen (awaiting root grant):

```bash
sg docker -c 'bash kernels/moe-task-order/run.sh task_order_1'
```

The launcher now holds the same shared GPU lock, checks vacancy, archives source bytes, validates the actual installed native source/capture on CPU and tracks only its owned subprocess group. It runs A1/B1/B2/A2 sequentially at L19/M1024/real routing. Every subprocess uses private `CUTE_DSL_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` and `FLASHINFER_WORKSPACE_BASE`, set before backend import. Runtime dispatcher/factory/native hashes are recorded. No installed source or persistent import state is changed; process exit removes the local factory replacement.

If S0 or ST is observed, the run writes applicability metadata then exits nonzero rather than time an identity permutation. Metadata includes actual T/S/G, long/short task counts and slice counts. Per-task durations are not measured by this lightweight harness; slice-count differences are a work hypothesis, not a predicted gain. Full-call graph timings and repeated numerical distributions are the measured quantities. CPU preflight passed with CUDA uninitialized and the known real capture hash. Reserve approximately3–8minutes for four isolated cold JIT namespaces; actual compile time is unmeasured. No GPU run has occurred.

## First applicability result

`results/task_order_1` terminated with exit1 at the intentional queue gate before any candidate execution. Real L19/M1024 had T231,S231,G2: all462 tasks were split/short and zero unsplit/long tasks. The proposed reorder is exactly identity there. GPU descriptors matched the CPU model, source verification passed and the GPU compute list was empty after cleanup. This is an applicability rejection, not a candidate performance result or kernel failure.

The locked launcher accepts an explicit second token-size argument1024/2048/4096 for a newly granted shape. Next requested screen is M2048; no retry has run yet. Previous source/executable snapshots and failure evidence remain intact.

### Measured queue applicability scan

`results/task_order_geometry_1` completed baseline-only checks, exit0, source verification passed and GPU-after empty. Real captured L19, tileM128/G2:

| M | physical T | split S | long tasks | short tasks | total tasks |
|---:|---:|---:|---:|---:|---:|
|2048|274|274|0|548|548|
|3072|339|339|0|678|678|
|4096|400|335|65|670|735|

Every GPU descriptor matched the CPU order model. M2048/3072 are also identity permutations. M4096 is the first observed mixed queue and the only requested next ABBA candidate. The scan measured no speedup or per-task duration. Full per-expert row counts are preserved in applicability JSONs. Next command awaiting grant: `sg docker -c 'bash kernels/moe-task-order/run.sh task_order_4096_1 4096'`.

### First mixed-queue ABBA result: speed lead, numerical investigation required

`results/task_order_4096_1` completed all four fresh-process runs, exit0; source verification passed, process group exited and GPU-after was empty. Same captured real L19/M4096 inputs, routes and scales; same exact packed descriptor multiset (T400/S335/G2,65long+670short).

| run | native full-call median us |
|---|---:|
|A1 baseline|434.976|
|B1 long-first|417.936|
|B2 long-first|417.648|
|A2 baseline|434.304|

Both candidate medians are below both baseline medians, about3.9% lower latency. This is one layer/input shape on hot repeated weights, not model-level performance evidence.

Each distribution below contains64 pairwise comparisons from eight outputs per run:

| comparison | relative-L2 min | median | max |
|---|---:|---:|---:|
|baseline–baseline|0.002520|0.002943|0.003390|
|candidate–candidate|0.002717|0.003087|0.003312|
|baseline–candidate|0.004787|0.004820|0.004862|

The entire observed cross-variant range is above baseline self-variation. Reordered BF16 atomic accumulation can explain a systematic change, but that has not established acceptable accuracy. Descriptor bijection does not prove output equivalence. Preserve the speed finding while withholding promotion: inspect per-expert/local results or compare both against an appropriate high-precision reference, then run current-stack NLL/task quality before any serving integration. No accuracy threshold was silently relaxed. No additional GPU work was launched after this screen.
