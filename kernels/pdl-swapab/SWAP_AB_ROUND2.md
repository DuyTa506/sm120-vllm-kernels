# Explicit swap-AB inventory and next controlled experiment

This is separate from the PDL experiment. The rowwise DeepGEMM adapter's
`not swap_ab` assertion is a local unverified-contract guard, **not** evidence
that native SM120 swap-AB is unavailable.

| Path | Supported contract / orientation | What was actually tested | Missing experiment |
|---|---|---|---|
| DeepGEMM native dense FP8×FP8 | `[M,K] @ [N,K].T`, UE8M0 block scales (host accepts K granularity 32/128), BF16 or FP32 row-contiguous output. API auto-swaps for M1–16, no accumulated C. It swaps data, scale tensors and recipes and writes through transposed output strides. | `deepgemm-sm120/bench_dense.py` tested automatic branch at M1/2/3/4/8/16 across Qwen N×K = 12288×2048, 2048×4096, 9216×2048, 248320×2048; old table reports kernel/config and FP32 dequantized reference. | Same M/N/K/bytes/scales/output, forced native off versus on. Existing cross-library timings are not this ablation. |
| DeepGEMM native symmetric packed FP4×FP4 | Same auto-swap gate. UE8M0 scaling, not checkpoint NVFP4 per16 E4M3 + globals. | Grouped MXFP4 proxy timings and native regressions; not a native NVFP4 checkpoint swap experiment. | Separate same-format FP4 off/on test after FP8 control; cannot claim direct checkpoint applicability. |
| DeepGEMM mixed FP8×FP4 / FP4×FP8 | Native low-level mixed modes exist, but dense API explicitly excludes mixed operands from auto swap; mixed K requires multiple of128. | Not established as a supported forced-swap serving contract. | Override must reject rather than bypass this gate. |
| DeepGEMM FP8 batched einsum | FP8 operands only, native SM120 branch auto-swaps M1–32 without C; TMA-aligned copies may be inserted. | Native library regressions, not a controlled Qwen projection ablation. | Lower priority: no identified Qwen dense projection BMM callsite. |
| Experimental same-byte rowwise FP8 binding | Added row/column scale epilogue currently asserts split-K=1, no swap, no C. | Actual rowwise adapter tests in separate DeepGEMM track. | Supporting swap requires mapping effective row scale to original column scale and vice versa, verifying transposed stores and any split-K reduction. This has not been implemented/tested. |
| FlashInfer CUTLASS native NVFP4 MoE | Current decoded tactics already use weight-major MMA orientation (historical 128×32 tile, weights on MMA M and tokens on N). Native per16 E4M3/global scales retained. | Existing fixed-FI serving and cold-MoE timings exercised this backend; they did not toggle orientation within the same kernel. | Compare a valid alternative tile/orientation only if its compiled kernel exists; don't relabel existing swap as a new patch. |
| FlashInfer B12x native NVFP4 | Current MMA pipeline places routed token tiles on M; direct-micro is CUDA-core code and has no MMA A/B orientation to swap. | Historical cold comparison changed backend, activation-scale convention and kernel structure. It does not isolate swap-AB. | A real B12x orientation change must transpose operand/scale layouts, route indexing and epilogues; larger kernel work, not an API switch. |

## Source evidence

- Native dense gate: `csrc/apis/sm120_gemm.hpp`, original source SHA256
  `4f9217df242d576147be5e8edaf668a2aa3721f304024ed12e42de86a9e8fbb0`.
- Native host dtype/recipe/output checks: `csrc/apis/gemm.hpp`; scale conversion
  rejects non-UE8M0 SM120 MMA scales in `csrc/apis/layout.hpp`.
- Native BMM gate: `csrc/apis/einsum.hpp` (`kSwapAbMMax=32`, FP8 checks).
- Local rowwise guard: `deepgemm-sm120/fused_src/csrc/jit_kernels/impls/`
  `sm120_fp8_fp4_gemm_1d1d.hpp`, immediately after config/split-K selection.
- Historical dense artifact and scope: [DeepGEMM README](../deepgemm-sm120/README.md),
  `bench_dense.py`, `results/dense*.jsonl`.
- Historical backend comparison: [bench_cold_moe.py](bench_cold_moe.py). It used
  unit activation scales and must not be treated as calibrated NVFP4 equivalence.

## Prepared on/off control

[prepare_swap_control.py](prepare_swap_control.py) produced
[swap_control/native_swap_override.patch](swap_control/native_swap_override.patch)
and [manifest.json](swap_control/manifest.json), without changing or building
another agent's DeepGEMM sources. The patch adds `DG_SM120_SWAP_AB_CONTROL` with
`auto`, `off`, `on`; forced on is rejected outside the original native eligibility
gate. `DG_SM120_SWAP_AB_TRACE` emits actual selected orientation and M/N/K before
capture, so an unpatched library cannot silently ignore the experiment setting.

First measurable experiment: native FP8, N×K=2048×4096 and12288×2048,
M=1/3/8/16. Same quantized activation/weight bytes, same packed UE8M0 scales,
row-major BF16 output, no C, PDL off. Compare forced off/on in alternating order
with a weight pool larger than L2, CUDA graph replay and independent changing-
input replays. Check both outputs against FP32 dequantized operands and each
other. Record actual orientation trace, generated tile/stage/split-K choices,
full extension/source hashes and per-repeat distributions. Heuristics may
choose different tiles/split-K after orientation changes; report the complete
native swap policy effect rather than claiming all other scheduling stayed fixed.

A passing result measures the native quantized mechanism. It is not a vLLM
speedup or rowwise/NVFP4 model integration. Only then decide whether to extend
the guarded rowwise epilogue or port a native NVFP4 variant. The forced override
compiled successfully in an isolated no-GPU container. Native GPU smoke and two
fresh-process matrices are now complete with all numerical/replay gates passing.
The binary SHA256 is `93488bde73201c38ffc670ff79ee14089052f5aa7a9fd0a396b603b232a3137c`;
[build evidence](swap_control/build_result.json) and `swap_control/build.log` preserve
the CPU build. The lease-aware launcher is `run_swap_ab_control.sh`; the first
queued smoke uses M=3, N×K=2048×4096 and twelve alternating repeats.

## Measured outcome

`swap_native_matrix_1` and reversed-cell `swap_native_matrix_2` both exit 0,
cleanup 0, with matching complete source fingerprints. See the README table and
[validated repeated report](results/swap_native_pair_1.json). Same-shape off/on
outputs are bit-identical in all sixteen comparisons. For N2048×K4096, M3/M8
repeat around 10–11% lower native latency, while M1/M16 magnitudes vary widely.
N12288×K2048 repeats show about 2.3–4.1% lower latency. Tile/stage policies change;
final template parameters show split-K4 for N2048×K4096 and split-K1 for
N12288×K2048 (both orientations). The original config trace prints pre-finalization
split-K and is retained with this correction. This is existing native auto-swap benefit, not a newly integrated
vLLM optimization. A rowwise epilogue port remains a separate implementation and
measurement task; no rowwise or calibrated NVFP4 support is implied.

## Concrete rowwise integration hypothesis (not implemented)

The current experimental `fp8_rowwise_gemm_nt_sm120` API asserts M≥32 and its
launcher rejects swap. A small-M port cannot simply exchange scale pointers:
the epilogue deliberately evaluates `(acc * token_scale) * channel_scale` with
round-to-nearest FP32 multiplies. Swapping effective row/column axes would turn
that into `(acc * channel_scale) * token_scale` and change rounding.

A separate owned prototype should exchange the operand tensors, unit UE8M0 grids
and effective row/column scale pointers, preserve the original multiply order in
the transposed-store branch, and retain split-K=1/no-accumulation guards. It must
also relax the API's M floor only for explicitly tested small-M shapes. Start
with slices M1/3/8/16 of captured actual GDN FP8 activation/weight bytes and their
row/channel scales; compare against CUTLASS and the FP32 scaled reference,
including zero/nonuniform/extreme scales and changed-input graph replay. This
would be a new contract-preserving candidate; the measured native policy alone
is not that implementation. Shared `deepgemm-sm120/fused_src` remains untouched.
