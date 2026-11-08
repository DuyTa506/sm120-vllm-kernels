# Native NVFP4 split prefill — isolated CPU preparation

Status: source/operand audit and fixed-buffer publisher+phase invocation implemented; CPU tests and offline publisher SM120 compilation pass. No GPU execution, timing, serving integration, or selection. Target is actual Qwen3.6-35B-A3B NVFP4 L19/M1024,4096, then L0/39; preserve captured expert IDs and weights.

## Source and concrete hypothesis

Experimental FlashInfer revision `fc4df7ee1a1edbd713ff1438c25f8de6b3e5d3af` contains native block16 E4M3/E2M1 phase kernels. Six local research sources were checked against the original fetch manifest (`source_audit.json`). The missing upstream `intrinsics.py` was fetched at that revision and its Git blob SHA verified against the archived tree. `upstream/` is immutable source evidence; `adapter/` changes only the intrinsic import namespace. Installed FI0.7.0.post1 and main checkout are untouched.

The hypothesis is to separate route/quantization, FC1+SiLU+requant, and FC2+scatter, reducing the monolithic shared-memory/register union. Phase1 uses23,040 bytes shared memory, phase2 uses13,824, M64×N128 compute, four warps and two stages. Both launch fixed-capacity two-CTA-per-SM grids. These are native `mma.sync ... kind::mxf4nvf4 ... scale_vec::4X ... m16n8k64 ... ue4m3`, not an MXFP4 proxy.

Retain M128 **source** tiles. Upstream explicitly warns M64 source multi-tile experts are known-wrong; M64 compute subdivisions do not relax that source restriction. Initial contract is K2048/I512/E256/top8, SiLU, shared input scale, no swap, no deterministic output and no ready-queue mode. A4 planner requires ModelOpt representation and must not be handed this checkpoint's compressed-tensors raw object.

## Effective operand contract

Use the existing `moe-nvfp4/moe_common.py` preparation and exact checkpoint scales after byte/decoded proof. CPU `check_operands_cpu.py` verifies all256 experts in each of L0/19/39: raw packed dimensions, positive global scales, equal gate/up global scales, exact [up,gate] decoded weight reordering, and literal phase F8_128x4 addresses versus the existing reshape/permute. Results are `results/operands_l{0,19,39}_cpu_v2.json`. This does **not** assert an actual GPU prepared tensor or complete arithmetic result has been validated. The earlier result file's descriptive global-scale convention was superseded by v2 after reading the pinned intrinsic; its byte proof was valid.

| Phase input | Adapter value and condition |
|---|---|
| w13 payload | Existing prepared `[E,2I,K/2]` bytes, up then gate; no additional permutation |
| down payload | Existing `[E,K,I/2]` packed bytes |
| SFB planes | Existing F8_128x4-swizzled E4M3 bytes; K/16 granularity |
| packed A/SFA | Shared token quantization with existing `ops.scaled_fp4_quant`, gathered to expert-major M128 domain; preserve quantized block scales unchanged |
| FC1 alpha | Existing prepared `g1=(1/weight_global_scale)*a1`; unit-scaled SFA requires a1 here |
| phase1 global_scale | Existing prepared `a2_gscale=1/a2`. Upstream helper takes a **multiplier**, unlike the installed wrapper's reciprocal-style public fc2_input_scale |
| phase2 down_alpha | Existing prepared `g2=(1/down_weight_global_scale)*a2` |
| FC2 output | Original topk weights and BF16 atomic scatter; accumulation order may differ |

Upstream comments describe its own SFA folding; they do not establish equivalence with our prepared operands. The multiplier direction and alpha fold must be checked by an intermediate dequantized reference on the first GPU correctness run. Do not substitute similarly named installed intrinsics without comparing their math: both packages expose `quantize_block_fp4` with different lower helper conventions.

## Reusable domain publisher

`domain.py` is a CPU reference for counts → prefix physical M128 tile bases → token/route-slot map → per-physical-tile expert/valid-row metadata. Phase metadata stride is I/128=4; it is **not** the existing packed deferred-task queue. Every logical routed contribution is represented once; tails use sentinel -1. It can serve split phases, grouped masked GEMM, and alternate task schedulers without changing router decisions. `test_domain.py` covers empty/exact/tail/multi-tile experts, duplicate rejection and complete scale-address bijections.

The device implementation has a stable ABI: counts[256], tile_bases[257], token_map/route_slot/weights[capacity_rows], packed_A[capacity_rows,K/2], swizzled_SFA[capacity_rows,K/16], task_expert/task_valid_rows[capacity_tiles*4]. Capacity can be conservatively `ceil(M*8/128)+255` physical tiles; active tile count is device-resident. Intermediate size is288 bytes per physical row (payload256+scales32), e.g.14.06MiB for400 L19 tiles. Captured L19/M4096 routes have400 tiles; never assume that for other layers or requests.

Implemented device path (`device_publisher.py`, `split_adapter.py`): count kernel, device prefix/publish, and shared-input pack/gather preserving exact routed slots, followed by fixed phase1/phase2 launches. Zero unused rows/scales before publishing; validate no missing/duplicate contributions. A prepared-domain compute-only correctness smoke is useful, but any claimed improvement must include route, quantization, intermediate traffic and output clearing. No host counts or routing reads inside timed/graph execution.

## Existing versus missing mechanisms

| Mechanism | Already present/evidence | Specific missing experiment |
|---|---|---|
| Persistent dynamic scheduling | Installed b12x has compact descriptors, atomic claim/prefetch, split-task occupancy heuristic; phase kernels themselves have persistent fixed grids | Separate route domain from both GEMM resource footprints; persistent scheduling alone is not new |
| Long-first/MegaMoE-style balancing | Existing b12x targets roughly four waves for M≤4096; tested long-first has identical full2048 contribution payloads but inconclusive serving pair | Reuse counts/tile domain to select task granularity; preserve actual routes and charge policy overhead |
| Masked grouped execution | Installed b12x already handles expert counts, valid rows and padded physical tiles | Feed the reusable domain to an actual block16-compatible grouped implementation, not DeepGEMM block32 UE8M0 proxy |
| Warp and copy pipeline | Installed native gated path has asynchronous copy/TMA pipeline mechanisms; split phases use cp.async with two stages | Measure separate phase register/shared footprint and stalls before changing stage/warp counts |
| DeepGEMM SM120 support | Actual compatible FP8 QKVZ runs and vLLM activation were proved in `deepgemm-sm120`; branch naming does not negate that evidence | Archived grouped FP4 tests were MXFP4; NVFP4 MegaMoE API there requires architecture major10. Transfer scheduler ideas while retaining native block16 math |
| xinyang-zhou/sm120_nvfp4_ops | Root is pinning/auditing this newly supplied repository | No source claims until pinned files are available; map its publisher/MMA/scale ABI into this table before reuse |

## Next correctness screen, before timing

1. L19/M1024 and4096 real captures: build source domain, prove IDs/weights coverage and decoded prepared weight/scale equivalence.
2. Compare phase1 quantized intermediate against a reference using the exact activation quantization, FC1 scalar and BF16 activation rounding. Compare phase2 weighted contributions before atomics when possible; use repeated baseline scatter noise separately.
3. Verify changed-input/changed-route CUDA graph replay and tails. Then L0/39.
4. Only after correctness, compare complete split route+quant+FC1+FC2 latency against current scalar native b12x with identical operands. Report phase and total costs. Existing installed monolithic path is the meaningful baseline.

CPU commands: `python3 kernels/b12x-split-prefill/test_domain.py`; run `check_operands_cpu.py` inside the existing image with `CUDA_VISIBLE_DEVICES=` and explicit checkpoint snapshot. Isolated phase imports passed with CUDA uninitialized. There is no GPU launcher yet: publisher/phase invocation is implemented but native phase compilation, actual GPU correctness, numerical reference and changed-route CUDA graph replay remain unvalidated.

## Fixed-buffer implementation and CPU evidence

`Workspace.allocate` allocates by fixed M and worst-case tile capacity before capture; all runtime routes/counts remain device-resident. `SplitCall.assert_contract` performs setup-only scale checks; `compile` constructs CuTe views/phase kernels before capture. `run` calls the same `torch.ops._C.scaled_fp4_quant.out` used by vLLM, then clears/publishes the routed domain and launches both phases on the current stream. It includes all clearing and quantization work; phase-only costs must not be presented as a full-call improvement. Current implementation intentionally favors a simple auditable publisher over speed: four Triton kernels and explicit clears, one pack CTA per routed slot.

The publisher supports `-1` inactive route sentinels, rejects out-of-range IDs and duplicate valid expert slots through a caller-owned device error counter (output is not trustworthy when that counter is nonzero). Correctness harness must read/check it outside the timed/captured call. Runtime metadata, payloads and tails are refreshed rather than cached from prior routes. Initial public adapter scope remains M1024/4096, top8/E256/K2048/I512; CPU interpreter uses smaller adversarial shapes for practical coverage.

`compile_publisher_cpu.py` compiled all four actual Triton kernels to SM120 PTX with CUDA hidden; hashes and compiler metadata are saved in `results/publisher_compile1/`. CPU interpreter tests execute the actual kernels and verify counts, slot coverage, payload/scale byte preservation and replay resets. Neither test proves GPU execution or CUDA graph correctness. `split_adapter.py` imports with CUDA uninitialized. No production backend or shared installed package was modified.

Root's `docs/DEEPGEMM_SCHEDULING_REVIEW.md` was checked against the pinned new repository: the grouped dispatcher uses grid=SM count, block384, per-call descriptor/metadata refresh; compute code requires256 math threads. This agrees with its E0–E7 distinction between mechanisms already present and meaningful new experiments. Our publisher does not import its FP16 intermediate/local-only scale semantics.

## Prepared numerical/capture harness (not GPU-run)

`numerical.py` consumes the exact saved L19/M4096 actual-routing capture and uses its first1024 tokens. `run.sh` starts one owned subprocess session inside the existing dev container under the shared GPU lock and an empty-GPU check; it never stops the persistent container. Snapshots include experiment source bytes, imported helper/runtime source hashes, vLLM extension hashes, exact capture hash, checkpoint config/index identity and image ID. Cleanup validates PID start ticks/group identity and verifies the entire owned process group is absent; post-run hashes and GPU vacancy are recorded even on failure.

After compilation and eager warmup, the harness validates every count, insertion rank, physical tile, routed (token,slot), payload byte and SFA byte. Physical row order may change; comparisons use canonical route mapping. It independently calls the existing input quantizer again and requires identical payload/scales. Full valid phase1 output is compared against FP64 input/weight-unit GEMM, calibrated g1, SiLU/up, explicit BF16 rounding and the same a2 quantizer. The full phase2 output is compared against a fixed expert-order FP64 reduction of **actual** produced intermediate values, exact raw down scales, and FP32 `(g2*route_weight)` factors. This separates phase1 quantization error from phase2/scatter error.

Initial exploratory gates are relative L2≤1% for dequantized phase1 intermediate and≤2% for phase2/atomic output, with finiteness required. These are broad bug screens, not production quality tolerance: FP64 versus FP32 accumulation, nonlinear quantization boundaries and BF16 atomic summation differ. Exact payload/scale mismatch counts, per-expert errors, global maximum error and full initial domain/output tensors are retained. Current scalar full-call repeat noise and CUTLASS comparisons are diagnostics, not a replacement for the phase oracle. No timer is present.

Only after the initial arithmetic screen passes, the graph test replays changed input and routes/weights, then tail sentinels, then all-empty routes, and finally restores the exact original input/IDs/weights to test recovery. It rechecks complete domain and arithmetic references, compares graph output to fresh eager output within the stated2% exploratory bound, and requires empty output exactly zero. Error counter must remain zero. No global-scale reconstruction or host routing is inside capture.

Exact proposed single-run command, **requires root GPU grant**:

```bash
sg docker -c 'bash kernels/b12x-split-prefill/run.sh split_prefill_numerical_l19_m1024_1'
```

Estimated5–10 minutes including first CuTe compilation; no benchmark timings. CPU preflight passed in `results/harness_preflight1/`; reference nibble/swizzle/tail tests and exact subprocess cleanup fixture passed. Native CuTe phase compilation, GPU arithmetic and capture remain unvalidated until this run succeeds.

Capture layer identity is pinned through the original applicability sidecar (layer19/tokens4096/real), with its SHA and exact counts checked against tensor IDs. The archive itself has no layer field; this limitation is explicit in `capture_manifest.json`. Tensor keys, shapes and dtypes are checked before launch.

## Numerical1 result and queued timing screen

`split_prefill_numerical_l19_m1024_1` exited0, all source/runtime11/capture checks passed, owned child/group absent and GPU empty. The actual domain contains8,192 routed rows and231 active M128 tiles (capacity319). Initial/restore phase1 reference differs at only1 of4,194,304 dequantized values, relative L2=9.14e-5; changed-input/route and tail cases are decoded-exact. Phase2 versus actual-mid FP64 gives relative L2=.00380–.00389. Graph/fresh-eager differences are .00179–.00190; all-empty is exactly zero and restoring original inputs passes. Scalar repeat=.00450, candidate/scalar=.00686 and candidate/CUTLASS=.02607 are diagnostic relative L2 values, not production-quality acceptance.

Raw phase1 payload mismatch372,497 bytes is not372,497 numerical errors. Scale bytes matched exactly; saved actual scales are nonzero and decoded values differ only once. E2M1 has only the signed-zero representation ambiguity at equal values, so at least372,496 mismatching bytes must be signed-zero-only. The expected reference payload was not saved, so this is a deduction from exact decoded/scale comparisons; direct per-byte classification is added to subsequent prechecks. Offline evidence is `signedzero_offline*.json` in that run's measurement directory. Candidate payload itself has no negative-zero nibbles.

`timing.py` is now prepared, **not run**. It reuses unchanged `split_adapter.py`/`device_publisher.py`, performs the same full-domain numerical check before and after timing, and directly counts signed-zero-only/canonical/zero-scale byte mismatches. Warmup, compilation, graph creation and references are outside event intervals. Two reversed orders are scalar→split→CUTLASS and CUTLASS→split→scalar, eight samples each,50 complete graph replays per sample. Quantization, publisher, clears and both phases are included; initial weight preparation and router/topk computation are excluded equally. No observer/reference runs inside timed replay.

Proposed one-grant command:

```bash
sg docker -c 'bash kernels/b12x-split-prefill/run.sh split_prefill_timing_l19_m1024_1 timing'
```

This measures one real layer/shape and retains exploratory numerical gates; it cannot establish a vLLM serving gain or select a production backend. The initial host launch failed before any child/GPU because CPU preflight had created a root-owned results parent; only that parent ownership was corrected and the failed attempt is preserved in `results/launcher_attempt1_failure.json`.

## Complete-call timing1 result (M1024)

`split_prefill_timing_l19_m1024_1` exited0 with pre/post arithmetic and exact-domain checks passing, source/runtime/capture verification clean, owned child/group absent and GPU empty. The split path is **unselected** at this shape:

| Timing order | Scalar median | Split median | CUTLASS median | Split vs scalar |
|---|---:|---:|---:|---:|
| scalar → split → CUTLASS | 313.661 µs | 369.213 µs | 327.209 µs | +17.71% |
| CUTLASS → split → scalar | 313.948 µs | 368.804 µs | 326.922 µs | +17.47% |

Across both orders, ranges are scalar313.303–314.204 µs, split368.497–369.541 µs and CUTLASS326.328–328.581 µs. Raw samples are retained; this is a consistent complete-call loss for this actual layer/shape, not rejection of every split implementation or larger prefill. The run did not serialize workspace.tile_m; source evidence establishes its shared-workspace M128 route, not a direct runtime field read. Future runs now record that field outside timing.

Direct pre/post classification confirms372,496 signed-zero-only bytes with positive identical scales, zero zero-scale mismatches, and exactly1 payload-byte mismatch after signed-zero canonicalization. This validates the earlier algebraic explanation; arithmetic differences remain reported separately.

CPU-prepared next commands, each requires its own root grant:

```bash
# M4096 complete call plus component diagnostic
sg docker -c 'bash kernels/b12x-split-prefill/run.sh split_prefill_timing_l19_m4096_1 timing 4096 1'
# M1024 complete-call repeat plus component diagnostic
sg docker -c 'bash kernels/b12x-split-prefill/run.sh split_prefill_components_l19_m1024_1 timing 1024 1'
```

`component_graphs.py` separately captures quantization, publisher+all clears, phase1 and phase2. Each event sample follows a complete valid call and output reset **outside** its measured interval. Phase2 measures no accumulating-history artifact; publisher includes its actual core clears. The component caches/state differ from contiguous execution, and these values must never be summed or presented as the full-call result. Two reversed component orders collect12 samples each. Full-call timing remains separately measured, with unchanged core adapter/publisher and references before/after.

## Initial split direction closed: both prefill shapes lose

The M4096 and M1024 component follow-ups both exited0, passed full pre/post numerical/domain gates, retained source/runtime/capture identity, and left owned process groups/GPU empty. `results/initial_split_summary.json` collects complete-call medians, raw ranges and scope limits across all three timing runs.

| Real L19 tokens | Scalar median, two orders | Split median, two orders | CUTLASS median, two orders |
|---|---:|---:|---:|
| 1024 repeat | 313.558 / 313.600 µs | 369.234 / 368.701 µs | 327.455 / 327.137 µs |
| 4096 | 433.664 / 434.176 µs | 639.959 / 639.078 µs | 625.336 / 627.220 µs |

Live scalar workspace reads now prove tileM128: M1024 has231 physical tiles/462 tasks; M4096 has400 tiles/735 tasks. Split loses approximately17.5–17.8% at1024 and47.2–47.6% at4096. These results do not support integration or an agentic run of this candidate.

Standalone component medians, **not additive full-call measurements**:

| Tokens | Quant | Publisher + clears | Phase1 | Phase2 |
|---|---:|---:|---:|---:|
| 1024 | 6.144 µs | 38.912 µs | 198.656 µs | 90.112–92.160 µs |
| 4096 | 14.336 µs | 86.016 µs | 301.056–301.088 µs | 215.040–216.048 µs |

The simple publisher is a real cost, but these diagnostics do not support claiming that fusing its clears alone fixes the complete-call gap. GEMM/epilogue phases remain substantial. Timers for short components are quantized, and standalone cache/preparation conditions differ from contiguous execution. No component is diagnosed as memory-, register-, or atomic-bound solely from this table.

The useful deliverables retained are a native block16-compatible phase ABI, actual-checkpoint numerical references, changed-route graph-safe tile publisher and independent zero-encoding classification. Any subsequent kernel must beat current scalar complete-call performance, preserve the calibrated factors and BF16 rounding/weighted contribution contract, and be a separate experiment. Improving current monolithic tile/workspace selection (already owned by the parent/other agent) is more immediately justified than promoting this split.
