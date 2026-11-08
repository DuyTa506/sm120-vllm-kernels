# SM120 candidate inventory and next falsifiable experiments

CPU-only audit, 2026-10-28. Starting point is e535d2e: Qwen3.6-35B-A3B-NVFP4-Fast,
vLLM0.32, SM120, TP1/MTP2, scalar MoE opt-in. No GPU experiment or runtime
modification was performed for this inventory. Historical numbers below are
not results on the newly composed stack. Root alone schedules GPU jobs.

## All fourteen directories

| Directory | Existing evidence | Remaining work on current stack |
|---|---|---|
| fp8-attn-glue | Exact fusion, independently checked; root patch03 | Already in next; profile residual layout/quant costs before proposing another change |
| gdn-conv-fusion | Exact fusion, independently checked; root patch05 | Already in next; retain as parent, not a new round-two achievement |
| moe-nvfp4 | Shared/exact/scalar implemented and full serving measured | Actual routing/expansion/setup fusion is not implemented; investigate current dispatch first |
| gdn-norm-fp8 | Exact isolated chain, state probe, full/focused serving; decode losses | Integration ablation below; do not equate a failed composition with exhausted fusion potential |
| decode-route-capture | Signed target/draft traces; fixed-FI controls | Output equivalence unresolved; traces suggest rejected draft suffix work but do not prove removable DRAM traffic |
| deepgemm-sm120 | Pinned SM120 tests, graph safety and proxy dense/grouped benchmarks | Dedicated audit must distinguish MXFP4 block32 from checkpoint NVFP4 block16; test new compatible scheduling/kernel hypothesis |
| pdl-swapab | Historical PDL microbenchmarks; CUTLASS already swaps AB | Dedicated audit must test actual producer/consumer coverage and current compiled integration; not just flip one launch flag |
| dense-fp8-gemm | Prefill sweep negative; skinny FP8 decode partly verified | Fix compile-key gating and launch/error checks; test on current scalar before any composition claim |
| decode-roofline | Old optimized-profile roofline; skinny BF16 saves at c1 | Fix demonstrated warm AOT-cache flag hazard, verify off→on and on→off, then current-stack c1/c4 |
| gdn-replay | Bit-exact shadow/state evidence; historical c64 +7% rate | Independent current-stack state/preemption verification and c48/c64 run; do not expect c1 improvement |
| fused-router | Micro wins at small M; integration draft GPU-unrun | Start M<=3 guarded path where tested IDs matched; M>=4 changes some top-k sets and is a separate numerical-policy question |
| cache-granularity | Native finer matching reduces redundant prefill; extra splits offset gains | New in-forward tail-checkpoint export to remove additional scheduler steps; exact state alignment is mandatory |
| session-cache | Exact-position reuse measured; eviction/capacity caveats | Test bounded retention/eviction policy only after native finer-match baseline; do not repeat unbounded cache recommendation |
| kv-offload | Measured idle-session capacity recovery, sizing caveats | Current-stack idle-pressure test only for relevant workload; not a generic kernel speedup |

## Hypothesis 1: separate immutable MoE metadata from replay-dependent setup

Evidence: installed FlashInfer source
`flashinfer/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh`
lines1346–1375 computes activation/output offsets alongside weight and scale
pointers. Lines1379–1407 contain the cacheability TODO and
`computeStridesTmaWarpSpecializedKernel`: one thread per expert, GPU dependency
synchronization, and fresh expert offsets. The call order is documented in
`kernels/fused-router/README.md`: quant → fused expert maps → expansion → setup
→ GEMM1 → activation/quant → GEMM2/finalize.

New implementation hypothesis: keep per-layer immutable weight/scale pointers
and constant strides in stable device buffers, while the existing invocation
updates problem M, activation/output offsets and finalize mappings. If setup
still requires a launch, combine its dynamic writes with expansion only after
proving the expansion grid covers all experts, including empty ones. A separate
initialization kernel may be needed for zero-active-expert cases; do not hide it
from timing.

Critical compatibility constraint: shared workspace is reused across forty
layers. Immutable metadata cannot be cached solely by scratch address. Key by
weight/scale storage identity, device, shape, dtype, tactic/layout and lifetime;
prevent captured graphs from pointing at another layer's descriptors. Never
cache token offsets or expert counts, and never add host synchronization to
obtain active-expert counts.

Smallest experiment: first trace actual scalar dispatch at M=3/12/48 and
1024/4096/8192. CUTLASS setup optimization directly targets scalar's <1024
fallback; prefill uses b12x, so this source alone cannot support a scalar-prefill
claim. If b12x has analogous setup, target that implementation separately.
Run a device-only metadata oracle on alternating layers and changed routings,
empty experts, all-empty input and CUDA-graph replay; compare every consumed
pointer/shape/stride. Then cold-weight AB/BA full-MoE timing, retaining both GEMMs
and all setup costs. Reject if eliminated work is absent from the serving path
or total call time does not improve.

## Hypothesis 2: isolate GDN integration cost from the fusion arithmetic

Current source: `kernels/gdn-norm-fp8/overlay/vllm/model_executor/layers/mamba/gdn/`
`gdn_norm_fp8_fusion.py`, `packed_norm_fp8` around lines190–235. Even the
`original` route runs the original packed core followed by a compiled
`_quant_out` into externally supplied buffers. The projection consumes a
`QuantizedActivation`; the entire integration adds a split op. Thus selecting
original arithmetic for decode does not restore the original compiled graph.

New experiment: introduce an experimental ablation that uses original packed
core plus compiled quant-out for *every* shape while retaining exactly the same
QA consumer/split configuration. Compare baseline → integration-only → fused
prefill. This separates graph/interface cost from the fused kernel's arithmetic
and avoids assuming launch overhead caused the previously observed losses.
Use scalar MoE unchanged in all three, fixed FI controls and distinct caches.

First tests: verify integration-only Q/S/projection output and private states
against current reference at actual/padded sizes; trace one real GDN layer at
M=3/12/48/192 and prefill1024/4096/8192. Record GPU launches and CPU/CUDA-graph
boundaries. If the integration-only case explains regression, prototype a
static shape-specialized graph that leaves decode's original core/projection
interface intact, rather than merely branching inside the new opaque op.
Mixed prefill/decode must retain the general path; graph capture and state
semantics gate any performance test. No new whole-model quality claim follows
from an exact single-layer check.

## Hypothesis 3: export the finer prompt-tail checkpoint without another engine step

Existing `kernels/cache-granularity/README.md` section “Not done / next steps”
identifies this as unimplemented. Native fine matching saves repeated prefill,
but optimized paths incur extra forward splits. This is a workload-level
prefill optimization rather than a new GEMM tile.

Source targets in vLLM: `MambaManager.allocate_new_blocks`,
`is_mamba_prefill_checkpoint_valid`, `compute_mamba_prefill_checkpoints`, and
scheduler `_mamba_block_aligned_split`. Reserve a valid checkpoint column and
export the tail state inside the existing forward even when the chunk does not
span two block columns. Scheduler, manager and worker must agree on eligibility
and decline together. No state-table mutation without a correctness oracle.

Smallest experiment: CPU allocation/metadata fixtures for hits inside a block,
CoW, chunk boundaries, MTP acceptance0/1/2, eviction/preemption/abort. GPU private
state comparison at the exact saved position and ±1-token fault calibration,
then a short two-turn probe measuring engine-step count and newly prefilled
tokens. Reject on any state mismatch or missing reduction in split count.
Only then measure repository short/long subsets. Historical 35–65ms split cost
is a hypothesis-sizing clue, not a promised saving on scalar/next.

## Execution order and decision rules

1. Dedicated DeepGEMM and PDL audits produce compatible, executable candidates.
   Root can schedule their smallest correctness/micro jobs while CPU inventory
   and integration preparation proceed. Old proxy studies are retained, not
   silently relabeled as current-stack tests.
2. Capture current-stack dispatch/cost for hypothesis1 and run GDN integration
   ablation for hypothesis2. Prefer an actual removed bottleneck over another
   broad tactic sweep. No concurrent GPU jobs.
3. Complete skinny BF16/FP8 compile-key fixes, test each independently then
   jointly. Their overlapping work with fused-router must not be double-counted.
4. Independently validate gdn-replay on high-concurrency decode and a narrowly
   guarded fused-router path; assess performance at their supported shapes.
5. Implement hypothesis3 only with a state oracle; revisit session-cache and
   offload for measured capacity/idle-session workloads, not universally.
6. Record each as implemented/unrun, numerically passed, micro win/loss,
   vLLM win/loss, or unresolved. Compose measured compatible wins and run final
   agentic only after that selection. This inventory is not a completion claim
   and does not imply the kernel design space is exhausted.

## Additional root hypotheses: novelty and implementation choice

These hypotheses were requested after the initial inventory. They are distinct
from copying the older directory conclusions.

**Implement first: a static per-layer/token-bucket CUTLASS/b12x policy.**
The existing `moe-nvfp4/README.md` section3 sweeps CUTLASS tactics at layer19;
it does not select different backends per layer using held-out real routes.
Current scalar `FlashInferB12xExperts.apply` branches on host-known
`hidden_states.shape[0]` and `_hybrid_min_tokens` around line490. The weight-load
hook has the actual layer object, so it can bind a persistent layer identifier
without inspecting GPU routing during forward. A frozen table lookup therefore
adds no GPU→CPU synchronization and is deterministic for graph capture.

First candidate deliberately keeps all M<1024 on CUTLASS. The scalar load-time
hook rejects `_hybrid_min_tokens<1024` around line267: lowering that flag silently
changes scalar eligibility and mixes in a different b12x kernel family. Do not
conflate that experiment with the first policy test. For M>=1024, test whether
selected layers/buckets should return to CUTLASS instead of current b12x.

Smallest useful calibration job:

1. Capture real hidden states/routes independently for layers0/19/39 and several
   representative prompts. Existing layer19 routes do not establish other
   layers' behavior. Separate calibration and held-out prompts before timing.
2. At M1024/1536/2048/4096/8192, benchmark complete `_apply_cutlass` versus scalar
   b12x with identical weights, top-k IDs/weights and activations. Preserve fixed
   CUTLASS tactics, cold-weight rotation, output copy/finalize cost and AB/BA
   order. Numerical reference checks precede timing for every case.
3. Form a static policy only where improvement survives held-out routing and
   timing order. If all >=1024 buckets favor b12x, the policy hypothesis is
   rejected without another full serving suite. Do not deploy a table trained
   on random/uniform routes or on the same prompts used to claim its gain.
4. Bind table identity and content hash before compilation; register it in the
   cache identity. Unknown layers/shapes fall back to the existing policy.
   Verify boundaries and warm-cache off→on/on→off before serving.

**Dual-output activation quantizer: new, but not a direct scalar-prefill drop-in.**
Routed b12x returns `expects_unquantized_inputs=True` and performs FC1 quantization
inside its wrapper/kernel. There is no external routed FP4 tensor that a shared
producer can currently replace. The simple vLLM-level prototype therefore
applies only to CUTLASS fallback, where `_apply_cutlass` explicitly calls
`ops.scaled_fp4_quant`. The shared expert path starts on an auxiliary stream.
A correct dual-output kernel must preserve each path's independent global scale,
block maxima, rounding, output layout and padded rows. It also needs an explicit
stream dependency before either consumer reads its output. Compare full main+
aux makespan, not the sum of two isolated quant kernels; it may lose overlap.
A prefill version would require a new b12x prequantized-input contract and is a
separate kernel implementation. No such implementation was found in this repo.

**Routing-skew-aware tile grouping: new scheduling policy, not a new tactic sweep.**
Historical layer19 CUTLASS tactic sweeps were already within ~1% of the selected
pair, and the DeepGEMM study reports many prefill tile waves with little tail
imbalance. A new grouping policy must demonstrate a different source of wasted
work: block padding, occupancy or scheduler metadata traversal, especially at
intermediate M and changing active-expert counts. Classifying or permuting
experts has a cost and must remain device-side under graphs. An implementation
that changes token-expert layout must restore exactly the required finalize
order; do not interpret changed reduction order as a free optimization. Compare
held-out actual skew, balanced routes, all-to-one/expert clusters, empty experts
and changing routes under replay. Profile first; reject if the scheduler is not
material in full-call time. This is lower priority than the static policy
because it needs a new scheduler and the old prefill evidence limits its likely
benefit at large M.
