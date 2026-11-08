# SM120 round2 execution ledger

The user explicitly rejected closure after the prior three-directory round. The active scope includes all existing kernel candidates, DeepGEMM, PDL/swap-AB and new hypotheses. Root develops hypotheses and reviews evidence; three agents own implementation and testing. Agentic is reserved for the eventual selected composition. Prior e535d2e results remain immutable; main stays unchanged during this round.

## Current assignments

- deepgemm_round2: same-operand rowwise FP8 adapter (DG FP32 accumulation plus original-scale epilogue), then compatible SM120 scheduling/mainloop opportunities. Old MXFP4 timing is not NVFP4 proof. Adapter CPU checks complete; GPU smoke queued.
- pdl_round2: actual vLLM Inductor PDL0/1 with isolated caches, explicit container propagation, fixed FI reference and generated code coverage. First baseline smoke pdl0_probe_1 granted the single GPU slot; no automatic second job.
- new_kernel_directions: full14-directory inventory and static layer/token-bucket backend calibration. Preserve below1024 fallback, use real weights/routes and held-out validation.

## Root hypotheses

1. Static per-layer/bucket dispatch may beat the global1024 threshold without forward host synchronization. Measure complete calls, not GEMMs alone.
2. Dual-output input quantization may reduce repeated activation reads while preserving each branch scale; internal b12x quant and aux-stream dependencies are real constraints.
3. Skew-aware expert work grouping may improve utilization, but include classification overhead and preserve finalize semantics; do not substitute another old tactic sweep.
4. GDN integration-only ablation can separate QA/split interface overhead from fused arithmetic before proposing further fusion.

These are hypotheses, not claimed gains. One GPU job at a time under the shared lock; every grant requires terminal state and owned cleanup before the next. All candidate dispositions must cite actual evidence, including untested/blocked/negative results. Finishing one subset does not complete the goal.

## System-level extension and review decisions

User requested directions beyond existing kernel ideas. CPU source audit is assigned
in `2026-10-28-system-directions.md`: prefill chunk selection, speculative depth
under prefill pressure/acceptance, and residual graph-padding work. Installed
vLLM already provides batch-size-based speculative depth and target-router
padding masks; duplicating them is not a new optimization. Dynamic depth zero
must not be treated as a no-speculation baseline without checking draft-prefill
execution. These directions remain unmeasured.

Root review of layer dispatch requested changed-input graph versus fresh eager
checks for each backend, not just graph-versus-graph agreement; hot-cache screen
labeling; helper hashes and failure exit status. The persistent dev container has
a readonly real capture while the host worktree path is an empty placeholder.
Any fresh runner must resolve this mount explicitly.

GPU order remains PDL0 diagnostic, DeepGEMM same-byte adapter smoke, PDL1
diagnostic, then layer calibration. PDL0 process1836544 was independently
verified live; four greedy scenarios validated and ten NLL requests collected.
Terminal cleanup/source/FI gates are still required before transferring the GPU.

## First terminal evidence

- `pdl0_probe_1`: exit0;23 source files verified; fixed FI shutdown132 entries
  preserved expected hash. Root independently observed process/container absent
  and empty compute query before lease transfer. This is baseline only.
- `rowwise_adapter_round2_smoke1`: exit1 before timings, absolute CUTLASS header
  symlinks unresolved under /src. Cleanup0. Preserved as failed infrastructure
  trial; launcher remounted unchanged source at expected /work/src.
- `rowwise_adapter_round2_smoke2`: exit0/cleanup0, root checked empty container
  and GPU state. QKVZ M32 fulladapter19.934us vs CUTLASS19.815us (slower0.6%);
  M64 fulladapter20.588us vs21.206us (faster2.9%). Two timing orders, one smoke.
  Sparse BF16 mismatches remain; sampled FP64 error matches baseline and changed
  input graph/eager checks pass. Synthetic operands on actual geometry, not
  checkpoint or serving evidence. Fused epilogue and larger prefills next.
- PDL1 granted after smoke2 terminal verification, same frozen driver/helper.
  Layer dispatch calibration remains next after PDL1; larger DG adapter follows.

CPU tests independently rerun by root: layer policy2/2, PDL provenance/artifact/
normalization7/7 pass. These tests do not establish GPU speed or model quality.

## Next completed gates and active queue

PDL1 diagnostic completed exit0 with23-source verification and unchanged fixed
FI132 entries. Root independently checked PID1853513 absent, exact owned
container absent and GPU compute query empty before granting `layer_dispatch_1`.
Both PDL modes completed four greedy cases and ten cold NLL requests. Mode1
current saved AOT artifacts contain dependency instructions; this establishes
code participation, not measured runtime overlap or attribution. Fresh Inductor
algorithm choices remain an audit item. NLL differences exist and are diagnostic,
not a quality pass; repeated same-mode controls are required for interpretation.

The next single-GPU lease is per-layer full experts calibration on real captured
inputs. DeepGEMM large-M adapter and fused epilogue smokes follow, then focused
PDL prefill/greedy pair in reverse mode order. No candidate is promoted yet.

Root CPU-audited existing GDN replay patch against current next build source and
pristine image envs/model-state: dry-run passes, exact hashes saved under
`kernels/gdn-replay/results/round2_composition_audit`. Agent preparing isolated
current-stack replay composition; dynamic MTP/replay combination remains untested.
MTP scheduling primary workload now uses repository prompts; instrumentation
will be separated from timing, and random-token stress is not model workload evidence.

## Calibration and fused epilogue evidence

`layer_dispatch_1` exited1 on its numerical gate, with all12 cells recorded.
CUTLASS was slower than scalar in both timing orders in every calibration and
held-out cell (layers0/19/39,M1024/2048). Cross-backend differences and nonexact
same-backend graph/eager comparisons require diagnosis; no gate was relaxed.
The frozen policy promotes no cell. Root checked exact child terminal state,
source verification and empty GPU before transferring the lease.

`rowwise_adapter_round2_prefill1` passed exploratory numerical checks but lost
all4 large-M cells. Full adapter paired median means: QKVZ1024/4096
108.49/523.51us versus CUTLASS90.28/319.49us; out1024/4096 37.46/129.13us versus
30.29/102.12us. Order drift is visible, but loss sign holds in both passes.
No original-adapter prefill promotion.

`rowwise_fused_round2_smoke1` passed, including bit-exact fused-versus-standalone
scale arithmetic. Fused QKVZ M32/M64 paired means18.703/19.077us versus
CUTLASS19.971/21.891us (~6.35%/~12.85% shorter). Synthetic operands only, one
smoke, not production evidence. Root verified exit0/cleanup0 and vacancy before
expansion grant `rowwise_fused_round2_qkvz_expand1` (M33/63/65/1024/4096).
Actual checkpoint/activation validation remains required before vLLM integration.

PDL first pair shows decode throughput+0.90/+1.19/-0.68/+4.29% at c1/4/16/64.
Three Inductor body groups selected different configs, and NLL differs, so neither
PDL-only performance attribution nor numerical equivalence is established.
Focused PDL reruns now wait for a fail-closed Inductor config pin or an explicitly
scoped producer/consumer experiment, avoiding repeated confounded comparisons.

## Narrowing the viable region; repeatability diagnosis

Fused QKVZ expansion passed exploratory numerical/replay checks and was exact
against the standalone adapter at M33/63/65/1024/4096. It still loses at1024/4096.
Medium map also passed: fused changes versus CUTLASS at M96/128/192/256/512 are
-7.69/-7.39/+35.39/+10.37/+2.60 percent latency. The useful synthetic region is
currently32..128; no blanket prefill replacement is justified. Root verified
both jobs' terminal exit0/cleanup0 and empty GPU. Capture-only real-model operands
`gdn_actual_capture_round2_1` is granted next; pristine eager/noMTP capture is
operator-contract evidence, not composed-stack serving performance.

Layer repeat-oracle1 failed in optional-input logging (None.shape), before useful
diagnosis; failure retained. Fixed repeat-oracle2 completed exit0 in separate
shared/private processes. Same-input eager-eager, graph-graph and graph-eager all
vary for both backends; scalar median relativeL2~0.0033–0.0035 and CUTLASS~0.0005–
0.001. CUTLASS tactics remain18/57 across modes. Private workspace does not remove
variation. Together with BF16 atomic-finalize source, this supports accumulation
order variability rather than a graph-specific or shared-workspace-only defect.
The separate cross-backend~2% residual is documented FP4 arithmetic difference;
no correctness gate was relaxed and no static dispatch policy promoted.

## Actual operands and PDL constructor recorder

`gdn_actual_capture_round2_1` completed exit0/cleanup0: runtime-bound QKVZ
operators from layers0/9/29, actual FP8 a/b and token/channel scales, preserved
strides, exact captured output, snapshot/input/source hashes. It uses pristine
image eager inference without MTP and is an operand-contract dataset only.
`gdn_actual_replay_round2_1` passed all21 cells (three layers, M32/33/63/64/65/96/128).
Fresh full1024 CUTLASS matches captured outputs exactly; maximum crossbackend
relativeL2 across original/changed slices is2.633e-5, sparse BF16 mismatches remain,
and changed-input graph replay matches eager. Paired operator timings show
6.59–14.05% lower latency than CUTLASS. Existing skinny FP8 candidate comparison
and actual serving effects remain pending; do not equate patch/b source presence
with the active CONFIGnext runtime dispatcher.

Root verified actual replay terminal and vacancy, then granted `pdl_record0_1`
(mode0, constructor record). It collects a NEW controlled config reference,
not a reconstruction of ambiguous historical autotuning winners. Startup+10NLL,
no throughput claim. Runner PID1889576 was independently verified live.
GDN replay shadow and skinny cache-transition experiments are CPU-prepared;
all warm-cache activation and runtime state claims await GPU evidence.

## Stronger baseline comparison and recorder repair

`gdn_actual_replay_round2_2` passed21 actual-operand cells against the existing
skinny FP8 candidate (2,4 forM32..64, CUTLASS fallback above64), in addition to
CUTLASS. Fused latency is lower than the faster comparator in all21 cells:
2.70–10.06% across measured shapes/layers. This does not establish that the old
skinny patch is active in CONFIGnext. Opt-in integration remains CPU work; actual
vLLM graph activation, NLL and task-level gains are unmeasured.

`pdl_record0_1` failed startup because real custom Triton metadata contains a
Triton BF16 dtype that was not JSON serializable. Failure retained; no partial
reference accepted. A typed allowlist encoder and exact failed-constructor CPU
regression were added. Root verified cleanup/GPU vacancy before granting
`pdl_record0_2`; no throughput conclusions from recorder runs.

New prefill experiment assigned under `kernels/moe-task-order`: reorder identical
native NVFP4 b12x descriptors long/full tasks first, split/short tasks last.
Existing queue already performs dynamic scheduling; novelty is ordering only.
Must retain descriptor multiset, expert choices, packed inputs/scales and local
math. Atomic BF16 finalize variability requires baseline self-repeat evidence
and task-quality checks; no bitwise-preserved reduction claim.

## Recorder reference and upcoming integration

`pdl_record0_2` completed exit0 with23 runtime-source checks, six pin-source
checks, FI132 invariants and18 constructor/first-launch identities. Canonical
reference has18 identities; saved AOT binding covers96 autotuner objects across
10 identities, with remaining8 verified through actual constructor/launch events.
This is a new controlled reference, not a performance or model-quality result.

`gdn_actual_replay_round2_buckets1` passed9 actual cells (three layers, M48/80/112),
with fused latency lower than the faster CUTLASS/skinny comparator by5.61–7.92%.
Conservative DG integration set now32/33/48/63/64/65/80/96/112/128; no broad range
or large-prefill promotion. Same-source flag0/1 runner, all30 eligible-layer
preparation and actual graph-kernel activity verification are being prepared.

Native task-order source review caught a relative-import bug before ABBA:
load candidate under the original package-qualified module name. Actual import
succeeded, but installed FlashInfer factory initializes CUDA at import; it must
be described as a device-enabled preflight, not CUDA-free CPU evidence.
Single GPU lease remains task_order_1, with vacancy/lock checks, then GDN shadow.

## Native task-order applicability

`task_order_1` stopped at its intended applicability gate (exit1), before any
candidate execution. Actual baseline queue at real layer19,M1024 hadT=S=231,
G=2,462short tasks and zero unsplit long tasks. Published descriptors matched
the CPU oracle. The proposed permutation is identity for this case, so there
is no optimization to time. Root checked child terminal/sourcecheck/GPUempty.
A baseline-only geometry scan atM2048/3072/4096 is granted next, recording actual
T/S/G/tile size/counts. Select a genuinely mixed queue before another ABBA pair;
do not relax the gate or report an identity reorder as a benchmark result.


## Root verification: task order, GDN shadow and next GPU lease

Actual native task-order ABBA at L19/M4096 completed: baseline 434.976/434.304 us,
candidate 417.936/417.648 us. Full-call latency is about 3.9% lower. Descriptor
multiset is unchanged (T400/S335/G2, 65 long and 670 short tasks). Root inspected
measurement/comparison.json: cross-variant relative-L2 median 0.004820 exceeds
baseline self-repeat median 0.002943 and candidate self-repeat 0.003087. This is
not a numerical or serving promotion; investigate accumulation order/reference.

GDN replay_r2_shadow is terminal exit1. Startup checked 2040 rows without reported
mismatches, but boundary coverage remained zero. CUDA graph capture failed at
37/63 with a 384 MiB allocation OOM in gdn_state_replay.py:222 (shadow_step).
This is incomplete diagnostic validation, not an accepted correctness result.
Root verified the owned container absent and GPU compute-app list empty after
cleanup. Agent investigates shadow memory use without relaxing boundary coverage.

Next exclusive GPU grant: pdl_pin0_1 mode0 with the frozen controlled reference.
No automatic mode1 or GDN retry. DeepGEMM serving runner CPU review passed;
root independently ran integration/test_trace_gate.py directly (exit0). Its
trace gate correlates captured candidate kernels to actual M48/64 replay launch
ranges. Initial prefill2048 is fallback nonregression, not candidate speedup.
DeepGEMM serving remains queued after the controlled PDL pair.

New CPU directions: installed humming-kernels 0.1.17 compatibility/full-call
comparison; explicit swap-AB source/experiment inventory; actual Qwen MTP log
reports eager rebuilding of FlashInfer draft metadata, requiring cost attribution
before optimization. Agentic remains reserved for the final selected composition.


## Pinned baseline completed; GDN diagnostic retry prepared

Root inspected pdl_pin0_1 exit0 and event validation:18 constructor identities,
18 first-launch identities. This remains a NLL/pin smoke, not throughput evidence.
After teardown root confirmed the container absent and a fresh nvidia-smi
compute-app query empty; granted only pdl_pin1_1 using the same frozen reference.
Pair AOT/config coverage and NLL comparison remain pending.

Root reviewed GDN fix against the failed-run source snapshot and independently
ran all9 round2 unittest cases (pass). Diagnostic snapshot collection is suspended
only inside Worker.compile_or_warm_up_model, restored in finally; preexisting or
unexpected pending records fail rather than being discarded. Native replay
registration/library preparation is still checked. Real-request shadow retains
all64 rows and intermediate checkpoint verification. Added workload guards require
32 completed boundary requests with2304 output tokens each, and128 accounted
stress requests including completion and abort coverage. No GPU retry yet.


## Pinned candidate terminal; shadow retry lease

pdl_pin1_1 completed exit0; root inspected terminal file and fresh GPU vacancy.
Strict paired AOT/config comparison is still pending. Ten NLL diagnostics were
collected; logprobs-2 differs markedly (pin0 0.428639, pin1 0.282102), so no
numerical-equivalence or performance claim is justified. Within-variant repeated
cold NLL is required in the next serving comparison to measure self-variation.

Root strengthened GDN metric evidence: absent/nonfinite/negative preemption
counters and a decreasing counter fail instead of treating missing data as zero.
The actual pin0 metrics expose a genuine initial zero counter. All11 round2 CPU
tests pass. GDN replay_r2_shadow2 receives the next exclusive GPU lease, with no
automatic preemption or timing continuation. DeepGEMM serving remains queued.


## Shadow2 client failure and fixed-length retry

replay_r2_shadow2 completed startup/capture and30-layer runtime guard, then exited1
before boundary requests: --random-range-ratio1 allowed zero input length and
was rejected by the installed sampler. There is no real-request correctness
coverage from this run. The original OOM did not recur during startup.
Root inspected the installed RandomDataset source: fixed lengths use ratio0.
Agent's CPU preflight executes the installed sampling functions and verifies all32
requests have input128/output2304 at ratio0. Root inspected revised run.sh and
confirmed shadow2 container absent plus GPU vacant. Granted only shadow3 with
explicit ratio0; owned container gdn-r2-replay_r2_shadow3, session73158, initPID1970682.
No automatic preemption/timing continuation.

Root independently ran moe-task-order-diagnostic/test_compare.py in the existing
container with CUDA hidden: coverage, duplicate, partial-column, tail and FP64
reference checks passed. GPU diagnostic compilation/execution remains untested.

### Humming exact-contract direction (CPU audit)

Isolated implementation: `kernels/humming-sm120/`. Installed Humming0.1.17 supports native SM120 NVFP4, but bare vLLM Humming MoE conversion lacks an explicit checkpoint activation schema. The isolated adapter now supplies CT local-dynamic E2M1/group16 calibration and disables fallback. All40-layer CPU audit (120 projection comparisons) confirms Humming reciprocal-then-amax equals actual FI activation-scale preprocessing; heterogeneous raw FC2 calibration is already globally reduced by the current FI baseline. It is not an additional Humming-only loss of calibration. Runtime GPU schema/dispatch/accuracy remains untested.

The linked Humming blog's two-GEMM figures exclude expert-call glue. Planned micro screen includes backend route packing, activation quantization, two GEMMs and output reduction, while excluding router GEMM/topK and initialization. Actual captured L19 routing, private caches, source snapshots, changed-input graph/eager and repeat distributions are prepared. No timing or quality win claimed; no GPU grant for Humming yet.


## Current-stack GDN shadow passed; DeepGEMM serving begins

replay_r2_shadow3 exited0. Root inspected coverage_gate.json and reran the saved
workload checker:869070 checked rows,630 intermediate-boundary rows, no numerical
or fatal errors, all32 boundary outputs exactly2304 tokens; stress91 completed,
37 aborted,0 failed. Post-stress probe saved12 cases/1406 tokens. Actual preemptions
were0, so this passes shadow only, not the separate preemption gate or timing.
Root confirmed owned container absent and fresh GPU process list empty. Next
exclusive grant is dg_serving_off1 via integration/run.sh with CACHE_FAMILY=
dg-r2-pair1. Candidate requires separate cleanup/verification/grant.


## DeepGEMM serving control completed

dg_serving_off1 exited0. Root inspected trace_check.json: live M48 full graph
replay observed, zero candidate fused kernels, gate passed. Agent reports all8
prefill and32decode requests completed plus four eligible-length NLL cases;
these are screening measurements, not a gain. Root independently confirmed
owned container absent and freshGPUempty, then granted dg_serving_on1 with the
same frozen integration sources and CACHE_FAMILY=dg-r2-pair1. Actual captured
candidate execution and paired numerical/runtime checks remain required.

PDL strict pair report now verifies18 shared constructor/launch identities and
12 model-source identities across16 wrappers, with explicit singleton/bundle
proof modes. NLL drift remains unresolved; next focused pair includes five cold
rounds, not a quality-equivalence claim. GDN preemption geometry guards passed13
CPU tests, but real spec-field JSON serialization is under review before launch.


## Native swap smoke passed; DG serving preparation corrected next

DG dg_serving_on1 exited1 before inference: unit-scale torch.ones inherited
vLLM default BF16 while native packed-UE8M0 metadata requires FP32. No candidate
serving result. Agent fixes explicit metadata dtype and audits vLLM's global
DeepGEMM PDL enablement versus microbenchmarks' default-disabled policy; new
same-source off/on pair required, original artifacts preserved.

swap_native_smoke_1 exited0/cleanup0, independently checked by root: M3/N2048/
K4096 forced off10.9002us versus on8.4094us (~22.85% lower), cold640MiB weight
pool,12 alternating repetitions. Paired output error is zero; both changed-input
graph replays pass FP32-dequantized-reference gates. Native configs also change
(tile/stages), so this is a native FP8 UE8M0 swap-policy effect, not isolated
orientation and not a Qwen/vLLM gain. Root verified GPUempty and granted only
swap_native_matrix_1: M1/3/8/16 at both2048x4096 and12288x2048,25repetitions.


## Native swap repeat and corrected serving pair

Both native swap matrices completed exit0/cleanup0 with bit-identical off/on
outputs in all16 comparisons and passing changed-input graph checks. QKVZ-shaped
N12288/K2048 improves2.3–4.1% across M1/3/8/16 and both runs. N2048/K4096 M3/M8
improves about10–11%, but M1/M16 non-swap times vary strongly across processes.
The existing auto-swap policy is useful; this is not a newly integrated patch.
An isolated rowwise-swap extension is being prepared with original token-scale
then channel-scale multiplication order preserved.

Root independently passed the actual DeepGEMM preparation BF16-default regression,
fixed-PDL policy test and trace evidence test. Granted fresh dg_serving_off2 with
CACHE_FAMILY=dg-r2-pair2, frozen corrected sources. Actual worker reports PDL off;
matched ON requires a separate grant after completion/cleanup. Root also reran
all9 PDL bundle/pair contract CPU tests successfully.


## DG live graph proof, unresolved serving confounds; MoE diagnostic active

dg_serving_on2 completed requests and trace but exited5: both worker shutdown
artifacts absent, FI server_shutdown verification failed without a logged Python
exception. Source hashes verified; root confirmed container absent/GPUempty.
Trace passes:450 fused captured nodes,390 correlated to13 actualM48 replays,
exactly30perreplay. This proves activation, not full lifecycle validity.

Short serving screen is mixed: decode throughput+0.68%, meanTPOT+4.43%, prefill
TTFT approximately flat. Eligible-length NLL shifts remain unresolved. The agent
mapped persisted compiler selections using installed cache-key functions:3 of10
common-source selected configs differ, including the backbone gate/FP8 reduction.
No causal DG speed/quality conclusion or promotion. Explicit post-workload audit
and common-producer pinning are CPU followups; other candidates get GPU next.

Root launched frozen task_order_diag_a1 baseline under sharedlock; session39128,
container process/PGID18218,start_ticks238588692. It records exact pre-atomic BF16
contributions on actual L19/M4096 and performs no performance timing. No candidate
or other GPU autochain is authorized.


## Task-order contribution proof and first Humming screen

Root diagnostic baseline/candidate tile0 calls completed exit0 with source/child
cleanup verified. Independent all-pairs report covers59198 canonical contributions:
payload bits and fixed-FP64 references are identical across both repeats of both
variants. Atomic outputs differ; this identifies reduction-order drift for the
first128 columns, not a full model-quality guarantee. Root full2048 baseline
run task_order_diag_full_a1 now active under the GPU lock, session74162.

Humming smoke1 stopped at missing standalone ForwardContext; actual NVFP4 schemas
and effective scales passed. A genuine vLLM context fixed the harness, verified
with CUDA hidden. Smoke2 completed exit0 and cleaned up: expert-call ABBA scalar
312.672/315.424us versusHumming287.280/286.944us. Cross-backend relativeL2 is
17–18%, so this is not a valid speedup/promotion. Add CUTLASS/reference and isolate
activation quantization/weight layout/intermediates before interpreting timing.

Tiny rowwise-swap smoke passed actual/changed/zero-scale graph checks and M0/17
rejection. Layer0/M3 AUTO swap17.77–17.81us remains about4.5% slower than existing
skinny17.00–17.04us, despite beating CUTLASS20.04–20.09us. A narrow64x16 policy
is CPU preparation only; persistent grid is188CTAs, while auto has96 logical
work tiles, so distinguish launched CTAs from useful first-wave work.


## Full-width task-order proof; preemption coverage attempt completed

Root task_order_diag_full_a1 and full_b1 both exited0 with GPU released.
Full2048-column cross comparison validates59198 contributions, zero payload-bit
mismatches and zero fixed-FP64 reference difference. Atomic output crossrelL2
0.004749; own-reference errors baseline0.004372 and candidate0.004038. This
identifies reduction-order drift in the actualL19/M4096 case, not general model
quality. Agent prepares uninstrumented serving screen next; no default promotion.

`replay_r2_preempt1` completed exit1 solely because actual preemption delta stayed0. KVBLOCKS50/maxmodel16384 and actual geometry/token guards passed. Numerical checks passed over908610 state rows and390 intermediate-boundary rows; all32 outputs2304 completed, stress91completed/37aborted/0failed, post-stress probe passed. Scheduler admitted roughly4 requests and queued others. Source verification/owned-container removal/fresh empty GPU were confirmed. This is negative preemption coverage, not a numerical kernel failure; timing gate remains unmet.

Next CPU-prepared pressure direction holds cap50 fixed and increases deterministic sequence growth:8requests/c8,128input/6656output, crossing2176,4352,6528-token boundaries. Record live allocation/admission/preemption metrics to test why the previous cap only queued requests. Actual preemptiondelta>0 and unchanged all-row/boundary/state/stress checks remain mandatory. No GPU grant for this longer-growth run yet.

## Humming RU/RN numerical isolation completed

Both humming_numerical_ru1 and rn1 exited0 with source/child cleanup and GPU released. RU FC1/FC2 payload+scale bits match the independentRU oracle exactly. RN FC1/FC2 bits match independentRN oracle exactly, and FC1 matches actualvLLM quantizer exactly. Raw weight nibble decoding matches; stage GEMM errors using exact decoded operands are~0.17% relativeL2, separating weight/layout math from quantization.

On actual L19/M1024, fullFP32-reference relativeL2: scalar14.510%, CUTLASS14.494%, RU16.328%, RN14.490%. RN crossdifference versusCUTLASS2.123%, versusscalar2.856%. No timing measured in these diagnostics; RU smoke speed cannot be transferred toRN. Remaining FC2 float32-activation versus BF16-cast quant comparison differs in0.879% payloadbytes/1.415% local-scale bytes. Do not silently add an activation cast.

CPU-prepared next route: isolated RN three-backend micro/correctness cases on captured L0/L19/L39 atM1024/4096, distinct private caches, actual source identities and changed-input checks. No serving integration or promotion until numerical/quality and measured RN-performance evidence supports it.

## Task-order first live control; harness shutdown investigation

`task_order_serving_a1` completed eight real4096-token cold requests (median
nonstream client latency94.482ms), four identical-token NLL repetitions and a
separate trace. Actual trace:40 native gated calls totaling17.292ms. Post-workload
readback validates retained4096×8 route rows, T405/S330/tasks735 and nonidentity
queue. Explicit post-workload FI132 audit passes. Sources verify and the owned
container/GPU are released, but terminalexit5 records missing `server_shutdown`.
Logs again show EngineCore SIGTERM followed by API-manager SIGTERM during resource
teardown. Investigate signaling only the verified API process rather than the
NSYS launcher process tree; keep strict shutdown verification and preserve this
failed run. Fresh controls are needed after any harness correction.

Baseline cold NLL is0.240264/0.219173/0.199898/0.191320. Root per-position audit
finds distributed differences, RMS logprob delta0.829–0.877 and maximum~18nats.
No state-leak or ordinary-atomic-noise cause has been established; do not attribute
any candidate quality delta before characterizing this baseline variation.

The separate rowwise64×16 smoke completedexit0 with numerical/graph checks but
19.62us latency,11.2–11.3% slower than AUTO and16.1–16.2% slower than skinny.
This L0/M3 policy is rejected without expanding the losing configuration.

RN L19/M1024 full-call screen `humming_rn_l19_m1024_1` completed exit0 with source verification, owned-child absence and empty GPU after query. Humming280.416/280.512us versus scalar315.168/314.048us (10.85% lower paired mean), CUTLASS327.424/327.408us. Independent FP32 references on both disjoint capture slices are comparable across all three backends; residual crossbackend differences remain above own replay noise. No serving/quality promotion. See Humming README for exact errors and timing exclusions. GDN actual allocator audit explains4 initial admissions using10 physical blocks/request, not conservative16 peak-budget blocks; longer-growth pressure remains queued with actual preemption gate unchanged.

RN L19/M4096 `humming_rn_l19_m4096_1` exited0 with sources verified, child absent, GPU empty. Negative performance: scalar433.680/433.936us versus Humming580.016/580.064us (~33.7% slower); CUTLASS635.488/635.424us. Independent FP32 error remains comparable across all3 outputs on both disjoint slices. Thus no global RN dispatch; M1024 win requires layer0/39 validation and later serving/quality checks. No further GPU launch without lease.
