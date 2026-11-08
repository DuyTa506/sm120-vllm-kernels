RN numerical diagnostic `humming_numerical_rn1` completed exit0 with source verification, child absence and empty GPU. FC1 payload/local-scale bytes match vLLM exactly; FC1/FC2 bytes match independent RN oracle exactly. Full FP32-reference relativeL2 is HummingRN14.4900%, CUTLASS14.4940%, scalar14.5102% versus HummingRU16.3285%. HummingRN-vs-CUTLASS2.1225%, vs-scalar2.8556%; both GEMM stages given exact quantized operands remain~0.17%. This validates the rounding explanation on this capture; no RN timing or model-quality result yet.

Remaining FC2 difference: comparing Humming float32 activation quantization with vLLM quantization after an explicit BF16 activation cast changes0.879% payload and1.415% scale bytes. That comparison separates an activation-rounding difference, but does not prove it explains every remaining full-output difference. An explicit FC2 cast would be a distinct experiment; it is not silently included in RN.

CPU-prepared RN screen now includes scalar and CUTLASS baselines with order scalar→CUTLASS→Humming→Humming→CUTLASS→scalar, changed-input graph/eager checks and repeated outputs. `rn_matrix_cpu.json` enumerates actual L0/L19/L39 captures at M1024/4096 and validates disjoint available row ranges. Each command needs its own explicit GPU lease. Do not inherit the prior RU8.6% timing result.

Numerical RU diagnostic `humming_numerical_ru1` completed exit0 with verified source/child cleanup and empty GPU. On real L19/M1024, FC1 **and** FC2 payload/local-scale bytes exactly match the independent RU oracle. FC1 versus vLLM/RN differs in51.1177% scale bytes and17.1633% payload bytes. Raw packed-weight nibble decode is exact; FC1/FC2 GEMMs using the exact decoded Humming quantized operands differ from FP32 by0.1720%/0.1659%, consistent with BF16 output rounding for these captured routes. This narrows the large cross-backend discrepancy to quantization rather than a gross weight-layout/GEMM error in this case.

Against the independent FP32 reference with original checkpoint weights and unquantized activations: scalar relativeL2=14.5103%, CUTLASS=14.4940%, HummingRU=16.3285%. Humming differs from both by17.895%; scalar versus CUTLASS is2.639%. This single-layer reference result is not a model-quality verdict. Private RN GPU diagnostic remains queued, not run; no timing claim or promotion follows from numerical diagnostics.

Latest GPU screen `humming_indexed_smoke2` completed exit0 with verified source stability, child absence and empty GPU process list. Real L19/M1024 full expert-call ABBA medians: scalar312.672/315.424µs; Humming287.280/286.944µs (about8.6% lower hot-layer time). **Not promoted:** cross-backend relative L2 is17.89% on capture rows0–1023 and17.27% on rows1024–2047, versus own-repeat medians0.39–0.43% scalar and0.25–0.34% Humming. Outputs finite, changed-input graph/eager discrepancies similar to own-repeat levels. These timings do not establish a valid inference gain; quantization/activation/weight preparation must be isolated against intermediate or FP32 references first.

First GPU smoke `humming_indexed_smoke1` exited1 at the first expert call because the standalone harness lacked vLLM ForwardContext. Cleanup verified child absence and an empty GPU process list; failed sources/results are retained. Weight conversion succeeded and runtime metadata confirms HummingIndexedExperts on SM120, native E2M1 inputs/weights, E4M3 group16 local scales, no FP16 accumulation, and exact baseline-effective calibrated secondary scales. There is no output-accuracy or timing result yet.

The harness now uses actual single-rank ForwardContext around apply. `check_context_cpu.py` validates the installed valid-shape method and context restoration with CUDA uninitialized. Corrected GPU retry awaits a separate lease.

# Humming SM120 compatibility audit (CPU source review, 2026-10-28)

Status: native NVFP4 weight support exists in installed humming-kernels 0.1.17; an isolated adapter now supplies the missing calibrated activation schema. CPU scale checks pass on all40 layers (120 projection checks, CUDA uninitialized). No GPU benchmark or speedup claim yet. Bare `--moe-backend humming` does not explicitly select the same W4A4 contract.

`compose.py` generates `generated/nvfp4.py` from the installed source snapshot. It supplies `CompressedTensorsInputSchema(dynamic="local", group_size=16, num_bits=4)`, disables schema fallback, and asserts converted secondary scales and input schemas after weight preparation. Generated code is not mounted into any active process yet.

Important checkpoint-to-runtime distinction: down-projection calibration is heterogeneous in the checkpoint, but current FLASHINFER_B12X and FLASHINFER_CUTLASS already compute `max(1/checkpoint_input_global_scale)` and repeat it across experts in `prepare_nvfp4_moe_layer_for_fi_or_cutlass`. Scalar FC1 optimization only changes storage after that aggregation. Humming CT input conversion performs the same reciprocal-then-amax. Therefore preserving this aggregation is the fair current-baseline comparison; preserving raw per-expert FC2 scales would be a separate numerical experiment. `audit_scales.py` executes the two installed scale-helper bodies on CPU, with hashes recorded in `checkpoint_scales_cpu.json`; this proves helper arithmetic, not GPU quantizer rounding or full-backend output.

The actual checkpoint is unsloth/Qwen3.6-35B-A3B-NVFP4-Fast revision `1c3f884bc99aac2524f6d49bcbac8c88401afd66`. Its compressed-tensors group1 uses packed E2M1 weights, group16 E4M3 local scales and tensor global scales; activations are group16 E2M1 with dynamic local quantization and calibrated tensor scales. Dense QKV/Z/out projections belong to a different FP8 group, so global linear+MoE replacement must not be the first experiment.

The [Humming announcement](https://vllm.ai/blog/2026-10-21-humming) motivates testing SM120/121 native NVFP4 and indexed/grouped scheduling. Its MoE plots measure gate/up plus down GEMMs, excluding the complete routing/quantization/reduction pipeline. Those plots do not establish full-call gains for this model.

## Actual installed source hooks

Paths below are relative to `/usr/local/lib/python3.12/dist-packages` inside `qwen-prefill-dev` (vLLM0.32).

- `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:425–459`: compressed-tensors weights are aliased to `weight_packed`, converted through Humming; generated config describes weight group16 only. Line451 supplies no input schema.
- `vllm/model_executor/layers/quantization/utils/humming/moe.py:570–582`: missing input schema comes from `VLLM_HUMMING_INPUT_QUANT_CONFIG` or an empty config; explicit TODO says checkpoint input config is not read. Isolated adapter now pins the exact CT input schema and secondary-scale source and asserts conversion. `allow_fallback=False` must stay explicit.
- `humming/schema/compressed_tensors.py`: NVFP4 conversion preserves packed weights and local E4M3 scales; global multiplier convention becomes reciprocal. Merged gate/up secondary scale handling depends on whether scales match; inspect actual tensors before declaring equivalent. Humming format repacking is initialization work, not per-call arithmetic.
- `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py:105`: single-GPU default is indexed; `VLLM_HUMMING_MOE_GEMM_TYPE=grouped` selects grouped contiguous. EP changes defaults, so record effective class.
- Same file indexed apply760–854 includes alignment (possibly separately for down), activation quantization, gate/up GEMM, fused activation/quantization, down GEMM and `moe_fused_mul_sum`. Grouped apply871–980 additionally uses scatter/unpermute. Full-call timing must include all of this.
- Same file quant support includes native NVFP4 static weights/dynamic activations; NVFP4 expects unquantized input and internally quantizes. Do not feed already b12x-repacked weights or double-quantized activations.

## Smallest fair experiment to prepare

1. CPU audit resolved weight/input schemas and original calibration tensors for L19, including gate/up secondary-scale equality. Construct from original checkpoint, not b12x layout; record package/source hashes and checkpoint metadata. Stop if native E2M1/group16/E4M3 activation contract cannot be preserved.
2. Isolated process per backend, real captured L19 input/routing, M1024/4096 plus one skinny decode cell. Baseline current scalar/b12x, candidates Humming indexed and grouped. Run eager/replay numerical checks, including same-backend repeat distribution because existing BF16 atomic finalize is nondeterministic; no relaxed gate or silent precision fallback.
3. Time complete expert calls including quantization/alignment/scatter/final reduction in ABBA order with private caches and fixed seeds. Report initialization/repacking separately. This is a hot-layer screen, not serving TTFT.
4. Only after full-call and quality gates, isolate `--moe-backend humming` in current scalar+next serving configuration while leaving linear backend/attention/MTP2 fixed. Then evaluate independent linear replacement, and only combine proven patches for final agentic measurement.

Pending: actual runtime adapter/dispatch proof; native accuracy; full-call harness and measured latency. These are additional experiments, not exhausted optimization space.

## Numerical follow-up prepared (not run)

A concrete local-rounding difference is source-identified. Native scalar b12x `moe_w4a16_fp4_helpers.py` quantize_block_fp4/fast use cvt_f32_to_e4m3, whose PTX is `cvt.rn.satfinite.e4m3x2.f32`. Humming `process_input/quantization.cuh:70–73` increments E4M3 encoding when nearest conversion falls below raw scale, implementing upward rounding. On actual captured L19/M1024 input, the CPU scale calculation changes67,001/131,072 scale bytes (51.12%); largest adjacent relative scale step12.5%. This does not prove which backend has better model/reference accuracy.

`numerical.py` adds current CUTLASS as a third full-call comparator and the repo's independent FP32 dequantized-weight reference, with TF32 disabled. It separately checks raw packed-weight nibble decode, FC1 input payload/local scales against the actual vLLM quantizer with linear scale layout, FC1/FC2 GEMM outputs against FP32 using exactly Humming's decoded quantized operands, and FC2 activation/quantization. An independent torch ties-to-even FP4 oracle compares both RU/RN local-scale variants. FC2 comparison with vLLM explicitly labels its required intermediate BF16 cast. The vLLM quantizer wrapper and installed `_C` binary are hashed; its C++ implementation source is not shipped in this image, so its actual rounding is tested empirically rather than attributed from an unrelated checkout.

Optional RN is isolated under `rounding/include_rn`: only the upward-increment line is removed. `rounding_mode.py` verifies original/private header hashes and switches Compiler's include path within that process. Compiler includes header path/mtime in its cache key, and launcher uses a unique private HUMMING_CACHE_DIR for every run. Installed upstream RU remains unchanged. No serving integration or speed claim follows automatically.

Queued commands, only with explicit GPU lease:

```
sg docker -c 'bash kernels/humming-sm120/run_micro.sh humming_numeric_ru1 indexed 1024 numerical_ru'
sg docker -c 'bash kernels/humming-sm120/run_micro.sh humming_numeric_rn1 indexed 1024 numerical_rn'
```

CPU ties-to-even/zero oracle checks and syntax validation pass. Neither numerical GPU run has been launched.

### RN full-call screen: L19 / M1024

`results/humming_rn_l19_m1024_1` completed exit0 with unchanged source hashes, no remaining owned child and empty post-run GPU compute list. Both disjoint captured slices use independent original-checkpoint FP32 reference with TF32 explicitly disabled immediately before each reference. Scalar/CUTLASS/Humming RN graph relative L2 to FP32 is respectively 0.145101/0.144940/0.144901 at offset0 and 0.141307/0.141207/0.141201 at offset1024. Own graph/eager noise is scalar0.00422/0.00409, Humming0.00317/0.00341, CUTLASS0; cross Humming/CUTLASS remains approximately0.0213 and is not explained by own replay noise alone.

Hot-layer full-call medians in scalar→CUTLASS→Humming→Humming→CUTLASS→scalar order:315.168,327.424,280.416,280.512,327.408,314.048 microseconds. RN is10.85% lower time than the paired scalar mean in this screen. This measures backend route packing, activation quantization, two GEMMs and reduction; excludes router GEMM/topK and weight preparation. Timing uses the second capture slice. It does not establish cold-weight, serving or model-quality gains. Further RN layers/shapes remain queued; no policy promoted.

### RN full-call screen: L19 / M4096 (negative)

`results/humming_rn_l19_m4096_1` completed exit0 with unchanged sources, owned child absent and empty post-run GPU compute list. Scalar433.680/433.936us, Humming RN580.016/580.064us, CUTLASS635.488/635.424us: RN is33.7% slower than current scalar in this larger prefill shape. A global Humming dispatch is unsupported.

FP32 reference relative L2 scalar/CUTLASS/Humming:0.136158/0.136009/0.136018 at offset0 and0.132317/0.132344/0.132278 at offset4096. Own graph/eager scalar0.00314/0.00246, Humming0.00173/0.00104, CUTLASSexact. Cross Humming/CUTLASS0.02035/0.01949 remains above own noise. Same full-call scope and hot-cache limitations as M1024; no numerical or model-quality promotion. Layer0/39 M1024 is the next useful expansion, rather than assuming larger shapes win.
