# DP4A expert reuse — isolated preparation

Stage1 local port targets Qwen H2048/I512/top8, NVFP4, M1–8. No GPU correctness or performance result yet. Upstream source and Apache2 license are retained; source SHA and implementation comparison are in [PLAN.md](PLAN.md).

`qf_moe_port.cu` exposes caller-owned workspace/output and a reuse switch. Both variants use identical DP4A quantization/math; reuse-on deduplicates experts across tokens, reuse-off computes each route separately. PDL is fixedoff. Two memset operations initialize output/counters per call and must be included in fullcall timing. Invalid IDs act as padding, including entirely invalid rows/batches. Valid expert IDs must be unique within each token. Workspaces must not overlap across concurrent streams/invocations and must outlive graphs; same-stream sequential replay may reuse them. The caller must provide correctly typed same-device contiguous tensors via prepare.py.

CPU evidence: check_contract.py verifies independent geometry/address/product invariants. check_checkpoint_cpu.py checks actual L19 all256 expert packed-weight reorder, every scale address, effective scales, and gate/up global equivalence against installed FI helpers; its swizzle oracle explicitly omits only the forced CUDA transfer. All pass with CUDA uninitialized. Source-structure tests pass, but do not establish CUDA numerical correctness or synchronization safety.

Build-only command (GPU hidden):

```
sg docker -c 'docker exec -e CUDA_VISIBLE_DEVICES= qwen-prefill-dev python3 /work/kernels/dp4a-expert-reuse/build_cpu.py'
```

Numerical-first GPU work awaits root review/grant. Include M1/2/**3**/4/8, actual automatic FI selection and forceddirect where supported. M3 is relevant to MTP2; strict routed_pairs<32 excludes M4/8 from automatic direct dispatch. No installed FI changes, vLLM integration or agentic measurement.

CPU-only sm120a build/import completed successfully; source/artifact identities are in build_manifest.json and exact compiler/link flags in build_cpu.log. CUDA remained uninitialized. This is build compatibility evidence only.

Reviewed-wrapper follow-up: exact ranks/dtypes/shapes are checked, including topk weights[T,8], byte-view swizzled scales and CUDA input. Every writable contiguous byte interval must be disjoint from inputs and all other workspace/output buffers, including shifted aliases. Three source tests and two CPU route-fixture tests pass. Port and separate instrumented diagnostic both rebuilt/imported with CUDA uninitialized; refreshed hashes are in build_manifest.json. Generated comments now describe the actual geometry.

Numerical smoke is prepared, **not executed**:

```
sg docker -c 'bash kernels/dp4a-expert-reuse/run_numerical.sh dp4a_numerical_m1_1 1'
```

Expected first-run compile/reference overhead: roughly2–5minutes, subject to FI compilation. Root must grant each run; later sizes3/4/8 are separate jobs (M2 also supported). Launcher takes the shared GPU lock, checks vacancy, snapshots sources/binaries, uses private compile caches, verifies sources and exact child absence, records terminal status and GPU list. It does not delete the shared dev container.

Diagnostic records own graph/eager equality, reuse-off/on equality, changed captured routing, fully/partially padded replay, two independent workspaces, malformed-shape/alias rejection, and independent FP32 reference residuals. Its source adds FC1 pre-BF16 and activation snapshots plus a quant16 probe; FC2 weighted partials are inspected before finalize. Compares quant16 bytes/scales with actual vLLM, FC1 against exact quantized operand FP32, FC2 quant with the same compiled helper, FC2 weighted GEMM with FP32. No timing data or promotion. Snapshot overhead means this build is unsuitable for future performance measurement.

Actual current scalar stack uses CUTLASS below1024 via its hybrid branch. Smoke separately labels current-stack CUTLASS, explicit FI-auto and FI-forced-direct; wraps kernel factory selection only for diagnostic trace and restores those hooks afterward. FI-auto's strict32 routed-pair threshold is not the serving stack's dispatch policy.

Reference precision is reset/asserted immediately before each full-output/FC1/FC2 reference and stored in the report. Both normal and instrumented builds must match exactly on each real slice, own graph/eager replay and padding; failures preserve measured comparisons and traceback in screen.json. Factory logs are explicitly Python selection evidence, not GPU kernel traces. Completion status distinguishes quant-exact versus quant-mismatch diagnostics; `correctness_promotion` remains false in either case. No inference of quantization correctness from finite outputs.

M1 numerical completed `dp4a_numerical_m1_1`, exit0, source guard passed, owned child absent and GPU empty. Quant16 payload/scales match actual vLLM; FC1/FC2 exact-operand FP32 residuals1.54e-7/1.20e-7, BF16 activation exact. Normal/diagnostic/reuse/own graph-eager comparisons exact; padding passed. Full-output residual versus FP32 remains dominated by quantization. No timing/promotion. Two workspaces were exercised sequentially, not concurrently.

M>1 harness now additionally forces eight shared experts0..7 with per-token slot permutations while retaining real captured inputs and normalized weights. It explicitly labels this synthetic route fixture, records multiplicities for both real slices and synthetic routes, and requires every hot expert serve M tokens. Same normal/diagnostic, reuseoff/on, graph/eager and FP32 comparisons apply. Last-fixture stage diagnosis is labelled synthetic when M>1. M1 original source/results snapshot is preserved.

M8 numerical `dp4a_numerical_m8_1` exited0 with source verification/child absence/GPU empty. Real slices had31/40unique experts and max multiplicity5/7; synthetic eight-hot fixture had8tokens per expert. All normal/diagnostic/reuse/own replay checks exact; quant16/FC2 bytes exact; FC1/FC2 exact-operand residuals1.49e-7/1.24e-7. No timing claim. FI's own JIT cache persisted in its default workspace in these numerical runs despite private CuTe/Inductor paths; timing launcher now uses the actual supported FLASHINFER_WORKSPACE_BASE and asserts its resolved path.

Prepared timing screen uses normal build only, M1/3/4/8, two real slices plus synthetic hot routes, two reversed backend orders and100graph replays per median. All DP4A output/counter clears, quantization, both GEMMs, activation and reduction remain inside graph; initial weights/layout/scales, router/topk and input copies are outside. **Hot-layer repeated-input screen**, not cold bandwidth or serving evidence.

FI plans create reciprocal scales, once-folded alphas, immutable cached weightviews and independent static workspaces before capture. Its existing runtime reciprocal-scale conversion stays inside fullcall. Preplanned outputs must match the numerical adapter before timing; this gate is exact and intentionally stops for investigation if a FI backend has nondeterministic reductions. Own graph-repeat/eager residuals are recorded separately. No claim of a kernel bug from that gate alone.

Review-only command, one shape per grant:

```
sg docker -c 'bash kernels/dp4a-expert-reuse/run_micro.sh dp4a_micro_m1_1 1'
```

Expected roughly2–5min initial FI compilation per new private cache. No timing GPU run yet. Factory selections are Python evidence, not a profiler kernel trace; no correctness/model-quality promotion.

First M1 hot event screen `dp4a_micro_m1_1` completed exit0 with exact preplannedFI parity, own replay parity and clean source/child/GPU checks. DP4AreuseON medians16.672/14.816us (real0) and18.528/16.800us (real64), versus currentCUTLASS about45.2–45.3us and calibratedFIdirect about24.8–25.0us. M1 cannot demonstrate cross-token reuse. Orderdrift is material; no serving or cold-bandwidth conclusion.

New explicitly-labelled `batched` timing mode uses8samples×50graph replays per backend/order, saving both total event spans and normalized per-call samples. Same numerical/setup gates remain. Review-only commands:

```
sg docker -c 'bash kernels/dp4a-expert-reuse/run_micro.sh dp4a_micro_m3_batched1 3 batched'
sg docker -c 'bash kernels/dp4a-expert-reuse/run_micro.sh dp4a_micro_m8_batched1 8 batched'
```

Each requires its own GPU lease; M4 boundary is a later separate run if needed. Port and diagnostic CUDA sources remain unchanged. See VLLM_ADAPTER_PREPARATION.md for a possible zero-copy binding to current scalar weights/SFs/alphas after positive M3 evidence.

Stronger FI comparator added before batched M3: `fi_direct_preplanned_multiplier` forces direct_micro, binds existing calibrated a1_gscale/a2_gscale arrays outside graph and uses `input_scales_are_reciprocal=False`. The cached weightviews retain the same once-folded g1/g2; `_weight_views` prevents another FC1 alpha fold in launch_sm120_moe. This bypasses only per-call constant reciprocal/where/copy overhead, preserving quantization. Existing reciprocal=True FI-auto/direct rows remain. No equivalent flag switch is assumed for the auto MMA path. Every fixture requires bitexact output parity with the previously diagnosed reciprocal=True numerical comparator before timing, including the new multiplier row. CPU guard rejects multiplier mode unless forceddirect. Two planning tests pass; no GPU measurement of this new comparator yet.

Source audit: installed moe_dispatch.py3479–3487 folds alpha only when `_weight_views is None`; direct path1631–1645 converts reciprocal scales only if the flag istrue, otherwise forwards multiplier tensors. Direct kernel2218–2220 consumes w1_alpha/input_gs/down_input_scale independently. The source snapshots are hash-bound in contract_cpu.json. This is an explicit best-preplanned direct comparison, not a claim the current public wrapper already performs that optimization.

### Strong comparator M8: diagnostic trial, FI-auto invalid

`results/dp4a_micro_m8_batched1` exited 0 with source verification clean,
owned child absent and final GPU list empty. Its old finite-only numerical gate
was insufficient: FI-auto selected `_get_static_kernel` and had FP32 relative
L2 **1.0** on every fixture despite exact parity with the numerical adapter.
Its ~33 us times are **invalid comparisons**. The entire trial remains diagnostic;
raw snapshots are preserved. The requested M4/M1 chain was stopped for review.
This does not identify a native FI kernel bug: static adapter/workspace applicability
is still under CPU investigation.

Other recorded M8 medians (two reverse orders, us; diagnostic evidence only):

| Fixture | DP4A off | DP4A on | Current CUTLASS | FI direct reciprocal | FI direct preplanned multiplier |
|---|---:|---:|---:|---:|---:|
| Real offset0 | 51.215 / 49.371 | 49.269 / 49.151 | 61.396 / 61.469 | 49.677 / 49.650 | 41.037 / 41.056 |
| Real offset64 | 53.452 / 53.144 | 57.113 / 56.726 | 67.814 / 67.814 | 49.261 / 49.303 | 41.038 / 41.044 |
| Synthetic eight-hot | 38.985 / 37.191 | 26.756 / 24.770 | 38.926 / 38.926 | 49.237 / 49.272 | 41.027 / 41.009 |

Prepared direct variants remained bitexact with the calibrated direct adapter;
DP4A reuse/own graph checks remained exact. FP32 errors for valid providers were
~0.134–0.144. Expert reuse benefits highly shared synthetic routing; the stronger
FI direct implementation is the practical lead on these real inputs. These are
hot repeated-layer measurements, not serving or model-quality results.

The revised `reference_gate.py` rejects zero output against nonzero independent
reference, nonfinite outputs, error >0.35, or error >1.25×CUTLASS error+0.01.
CUTLASS must itself pass the absolute ceiling. These are broad sanity checks,
not numerical-equivalence or quality tolerances. Five CPU fixtures test these
rules; no thresholds were relaxed to admit the invalid static comparator.

Pending focused command (requires root GPU lease):

```bash
sg docker -c 'bash kernels/dp4a-expert-reuse/run_micro.sh dp4a_micro_m1_batched_checks1 1 batched direct_checks'
```

The optional checks exercise native `-1` routes, restore changed inputs, revisit
M1/3/4/8 graphs with separate persistent epoch/workspace buffers, and compare each
graph with fresh eager and FP32 references. They are sequential isolation tests,
not concurrent workspace reuse. A CUDA profiler trace follows numerical gates
and precedes timing; constructor traces alone are not kernel execution proof.

**Raw-native padding is unsupported; safe adapter now prepared, pending review.**
Installed direct FC1 reads `w1_alphas[eid]`, `input_gs[eid]` and
`down_input_scale[eid]` without a negative-ID check (pinned
`sources/moe_direct_micro_kernel.py:2206`). FC2 similarly reads negative expert
weights/scales (`:1205`, and the M1 specialization `:766`). Raw `-1` routes must
not be executed on this path. The pending test now uses private sanitized operands for every provider;
raw-invalid native invocation remains prohibited. See `DIRECT_PADDING_AUDIT.md`. `direct_native.py`
prepares an explicit launch handle without global dispatch mutation, but does
not itself change this upstream valid-ID-only contract.
