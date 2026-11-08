# DeepGEMM `nv_dev` SM120 kernels vs our stack (reference study, nothing integrated)

Question: can DeepGEMM's native SM120 kernels (branch `origin/nv_dev`, commit b64107f) or their scheduling ideas beat what
vLLM v0.32.0 runs for `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` on one RTX PRO 6000 Blackwell (SM120, 188 SMs), and are they CUDA-graph safe?
Everything here is a benchmark; no DeepGEMM adapter is integrated into vLLM. Round 2 uses the existing compiled study source read-only and stores separate evidence.

## Short answer

* **The original proxy study did not establish an integration win.** Its quantization contracts differ from the deployed model; the round-2 experiment below reopens the compatible FP8 path. Original dense FP8: our shipped skinny kernel (`kernels/dense-fp8-gemm`) is faster
  than DeepGEMM at M <= 16 on all four decode shapes (DeepGEMM only wins by ~1.2 us on gdn_qkvz at M = 32-64), and at prefill (M = 8192) DeepGEMM is 6-8 % *slower* than vLLM's CUTLASS (628-637 vs 674-677 TFLOPS).
* **Grouped GEMM, prefill (real 8k routing, hottest expert 6007 rows):** DeepGEMM's FP4 kernel (MXF4, a proxy, see below) needs 690-712 us for
  GEMM1+GEMM2, our FlashInfer CUTLASS NVFP4 GEMMs need 572 us (cold weights): DeepGEMM is 20-24 % slower, FP8 is 1017-1038 us. Skew costs
  DeepGEMM only through BLOCK_M padding (x1.27 rows); its scheduler has no skew handling (static persistent round-robin of equal tiles).
* **Grouped GEMM, decode:** the one real difference. DeepGEMM's BLOCK_M=32 FP4 kernels are 9-12 % faster than our CUTLASS GEMMs at 12-192 tokens
  and 45 % faster at 3 tokens (18.5 vs 33.5 us per layer, GEMM kernels only). About 6 points of that are the smaller MXF4 scale format (UE8M0 per 32 vs E4M3 per 16 = 5.6 %
  fewer bytes). The remainder is **not** explained by tile size (forcing BLOCK_N 64 -> 128 changes nothing) and I did not find the cause.
  It cannot be taken as a saving for NVFP4: no NVFP4 kernel was run.
* **CUDA graphs: issue #444 does not reproduce** on b64107f, which already contains PR #447 (merged into `nv_dev` on 2026-09-23 as a23e1f0, it is not "unmerged").
  14 configurations x 1000 replays with 48 different routings (empty experts, one/eight hot experts, different totals, all-empty): no error,
  every replay bitwise equal to an eager call. DeepGEMM's own SM120 suite passes (73/73 + 342/342).
* Original-study recommendation: no direct port on that evidence alone. A remaining follow-up is a profile (ncu) of our FlashInfer GEMM1 kernel at 3-12 tokens against
  DeepGEMM's BM=32 kernel to find where the ~7 us per GEMM at 3 tokens goes. Upper bound if all of it were recoverable: 15 us x 40 layers = 0.6 ms of a 6.5 ms c=1 step; not demonstrated.

## Round 2: same-byte row/token and channel FP8 adapter (2026-10-28)

Unlike the original dense proxy, this experiment gives both implementations identical FP8 bytes and original arbitrary FP32 token/channel scales. DeepGEMM receives unit packed UE8M0 scales and produces FP32 scratch; a fused standalone epilogue applies the original scales and casts once to BF16. Both paths use preallocated BF16 output, so output allocation is excluded. Scale preparation is also excluded. These are synthetic Qwen-shaped inputs, not checkpoint activations or a vLLM integration.

The granted smoke used GDN QKVZ N=12288,K=2048, 11 weight copies (264 MiB, above L2), CUDA graphs, and two passes in opposite order, 30 replays each. Numbers below average the two pass medians; raw medians/min/max are retained.

| M | CUTLASS us | DG FP32 core us | Full adapter us | Adapter change vs CUTLASS | Initial relative L2 | Initial BF16 mismatches |
|---|---:|---:|---:|---:|---:|---:|
| 32 | 19.815 | 18.403 | 19.934 | 0.60% slower | 5.58e-6 | 2 / 393216 |
| 64 | 21.206 | 18.679 | 20.588 | 2.92% faster | 1.42e-5 | 2 / 786432 |

Sampled FP64 reference errors matched CUTLASS for both shapes (relative L2 0.001547 and 0.001664). Three mutations of both activation bytes and token scales replayed bitwise equal to eager adapter output, with fresh CUTLASS comparisons finite and relative L2 <=2.60e-5. The harness's <=1% screening threshold is **not production numerical acceptance**. Results are not bitwise identical to CUTLASS.

**Conclusion:** the standalone adapter has a small M=64 microbenchmark gain and no M=32 gain. Its extra scratch/epilogue costs approximately 1.53/1.91 us relative to the DG core. This motivates testing a fused scale epilogue; it does not yet justify deployment, establish a prefill win, or beat the existing composed kernel configuration. Next gates are independent repetition, prefill shapes, comparison to the existing skinny candidate, representative numerical checks, and measured vLLM task effect. Agentic evaluation remains for the final composed configuration.

Evidence: [successful smoke2](results/rowwise_adapter_round2_smoke2/results.json), [log](results/rowwise_adapter_round2_smoke2/run.log), and [plan/contract audit](ROUND2_PLAN.md). Launcher exit=0, cleanup=0; exact owned container absence and empty postrun compute-process list were recorded, and all three harness source hashes passed postrun verification. The launcher snapshots the actual scripts, image ID, source revision/diff and isolated caches.

Provenance caveat: `source_revision.txt` in smoke1/2 reports the enclosing repository revision because this source was originally unpacked with `git archive`; it does **not** prove the DeepGEMM revision. `b64107f...` is the intended study pin, not a newly verified tree identity. Later runs fingerprint actual source/header/extension bytes and label the enclosing revision separately. The isolated fused experiment exports [its patch](patches/rowwise_epilogue_round2.patch) and a byte fingerprint of the actual original source, including the loaded extension.

The first [smoke1](results/rowwise_adapter_round2_smoke1/results.json) failed **before measurement**, because legacy source include symlinks point to `/work/src/third-party/cutlass`, while the first launcher mounted it at `/src`. The corrected launcher preserves `/work/src` read-only and mounts the frozen harness separately at `/harness`; a CPU-only container check confirmed headers readable. Failed evidence is retained unchanged; exit=1, cleanup=0, no lingering container or GPU compute process.

Reproduce with a fresh result tag after reserving the shared GPU lock:

```bash
./run_rowwise_adapter.sh NEW_TAG --shapes gdn_qkvz --ms 32,64 --iters 30
```

### Fused native row/channel epilogue and boundary results

The isolated [five-file patch](patches/rowwise_epilogue_round2.patch) adds coordinate-aware scaling inside the SM120 accumulator epilogue, removing the standalone FP32 scratch round-trip. The new API rejects split-K, swapped operands, accumulation, M<32, noncontiguous operands and non-BF16 output. It multiplies FP32 `(acc * sa) * sb` explicitly before BF16 conversion. The isolated host extension built successfully; offline NVCC compiled both TMA and direct-store template variants without accessing GPU. Runtime-selected kernels in these measurements use swizzle-CD=128 (TMA store), so direct-store numerical execution remains untested.

Actual GPU [smoke](results/rowwise_fused_round2_smoke1/results.json) and [boundary/prefill expansion](results/rowwise_fused_round2_qkvz_expand1/results.json), QKVZ N=12288,K=2048:

| M | CUTLASS us | Fused DG us | Fused change vs CUTLASS |
|---|---:|---:|---:|
| 32 | 19.971 | 18.703 | 6.35% faster |
| 33 | 21.290 | 18.599 | 12.64% faster |
| 63 | 21.886 | 19.283 | 11.89% faster |
| 64 | 21.891 | 19.077 | 12.85% faster |
| 65 | 21.817 | 20.093 | 7.90% faster |
| 1024 | 90.732 | 95.077 | 4.79% slower |
| 4096 | 318.126 | 360.638 | 13.36% slower |

All seven initial outputs and all three changed-input/scale checks per case were **bitwise equal to the standalone adapter**, while retaining the separate CUTLASS rounding differences. Changed-input graph replay matched eager execution. Both jobs exited0, cleanup0, with unchanged source fingerprints and empty postrun compute/container observations. These are two timing orders within one job per shape, not independent repeated runs or actual checkpoint validation. Small-M gains are promising; large prefill does not win. The current skinny dispatch and actual captured operands must be compared before vLLM promotion.

The subsequent [medium-M map](results/rowwise_fused_round2_qkvz_medium1/results.json) narrows the promising geometry: M96=21.137→19.512us (7.69% faster), M128=22.073→20.441us (7.39% faster), M192=24.886→33.693us (35.39% slower), M256=30.755→33.943us (10.37% slower), and M512=51.818→53.163us (2.60% slower). All numerical/replay screens passed, including exact fused-versus-standalone comparisons. This suggests studying a **32–128-row QKVZ dispatch region**, not replacing prefill GEMMs globally. Exact boundaries, independent repetitions, captured operands and existing-stack comparisons remain required.

The separate [original adapter prefill screen](results/rowwise_adapter_round2_prefill1/results.json) also passed numerical/replay checks but lost all four cells:

| shape | M | CUTLASS us | DG FP32 core us | Full standalone adapter us |
|---|---:|---:|---:|---:|
| QKVZ | 1024 | 90.278 | 95.580 | 108.487 |
| QKVZ | 4096 | 319.489 | 380.139 | 523.505 |
| GDN output | 1024 | 30.290 | 33.319 | 37.456 |
| GDN output | 4096 | 102.117 | 118.287 | 129.125 |

Timing order/clock drift was visible (e.g. QKVZ4096 CUTLASS302.5→336.4us); the standalone loss remained in both passes. These averages describe the observed burst runs and should not be treated as stable sustained throughput. No prefill or agentic improvement is claimed.

### Actual runtime operands: operator validation passed

A separate [capture-only vLLM run](results/gdn_actual_capture_round2_1/results.json) loaded snapshot `1c3f884bc99aac2524f6d49bcbac8c88401afd66` and consumed 1024 tokens of the repository's real-text prompt. This was the pristine image in eager/no-MTP mode, **not the composed scalar+next serving configuration**. After initialization, a capture wrapper identified actual GDN QKVZ modules by runtime weight pointer at layers0/9/29 and saved exact FP8 A/B bytes, token/channel scales and baseline outputs, preserving shapes/strides. Runtime source copies, image ID, checkpoint config/index hashes, prompt hashes and tensor/artifact hashes are recorded. Whole checkpoint shard files were not separately hashed; exact runtime weight tensor bytes were.

[Actual replay1](results/gdn_actual_replay_round2_1/results.json) passed all21 layer/M cases for M32/33/63/64/65/96/128. Fresh CUTLASS on the full captured1024 rows first matched captured output **bitwise** for all three layers. Fused versus fresh CUTLASS relative L2, including changed actual row slices, was at most2.633e-5; initial outputs had at most13 BF16 mismatches. Sampled FP64 reference errors equaled baseline. Changed actual activation bytes/scales replayed exactly between CUDA graph and eager fused execution.

Every cell was faster than CUTLASS in the measured two-order replay, by6.59–14.05%. This validates the operator contract on actual model operands and repeats the synthetic gain, but is not an end-to-end or NLL result. The existing repository skinny candidate is being compared separately; its presence in a source directory does not prove it is active in `CONFIGnext`. No serving integration is claimed yet.

Both capture and replay exited0/cleanup0 with unchanged fingerprints and empty postrun owned-container/GPU-process observations. Captured tensor `.pt` files are local ignored artifacts; JSON manifests preserve their hashes.

### Strongest existing candidate comparison

[Actual replay2](results/gdn_actual_replay_round2_2/results.json) compared all21 cells against both CUTLASS and the exact existing repository skinny kernel (original source/flags, `(2,4)` configuration for M32–64, CUTLASS fallback above64). Source, wrapper and compiled extension hashes are in `skinny_manifest` in the result. This is a candidate comparison, not evidence that skinny is active in the current serving stack.

All21 existing-candidate numerical checks executed; maximum relative L2 versus CUTLASS was8.454e-5. Fused DG beat the **faster measured comparator in each cell**: M32 by2.70–2.89%, M33 by5.97–7.35%, M63 by8.82–9.38%, M64 by9.03–10.06%, M65 by7.62–7.98%, M96 by7.35–7.85%, and M128 by6.54–7.00%. Replay2 exited0/cleanup0 with verified artifact hashes and no lingering owned container/GPU process.

An opt-in serving integration is prepared under `integration/`, but has not run in vLLM yet. It registers its flag in compile factors, prebuilds unit scale metadata/kernel configurations during weight loading, then uses an opaque custom op for a conservative exact verified-M set. It composes DG→existing skinny→CUTLASS only where dtype/layout/scale checks allow; other shapes and bias use the original path. The subsequent [actual bucket replay](results/gdn_actual_replay_round2_buckets1/results.json) passed9/9cells at M48/80/112. Fused gains over the best comparator were5.61–6.39%,7.27–7.92%,6.95–7.48%, respectively; these buckets are now in the conservative integration set. An intended c16/MTP test still must prove the live GPU path rather than infer it from the flag.

## What was run, and what the numbers mean

* Environment: `vllm/vllm-openai:v0.32.0` + `libdw-dev` (saved as the local docker image `dg-sm120-env:1`; torch 2.13+cu130, nvcc 13.0.88, which accepts `sm_120f`),
  DeepGEMM from `git archive origin/nv_dev` plus the two submodules at the pinned commits (CUTLASS f3fde58, DeepJIT e5bdee2), see `setup.sh`. The
  `third-party` submodules were missing in the local clone and were fetched from GitHub. One study patch (`patches/force_bm32_block_n.patch`, env var
  `DG_FORCE_BM32_BN`) was applied to the copy for the BLOCK_N experiment only; every other number is the unpatched heuristic.
* Timing: CUDA-graph replays (median of 30 replays), weights rotated through distinct copies (dense: >= 640 MB, MoE decode: 6 copies, MoE prefill: 2 copies) so the
  128 MB L2 cannot serve them; `kernel-only` columns are CUPTI device times of eager calls with the same rotation. GPU was otherwise idle when these
  ran (SM clock 2.6-2.9 GHz, short bursts), so absolute TFLOPS are **burst** numbers, not the 611 TFLOPS / 1895 MHz / 591 W sustained figure in
  `kernels/dense-fp8-gemm/README.md`. Run-to-run: median 0.7 % between two dense passes (max 27 % on one tiny case).
* **Quantization contracts differ, so DeepGEMM numbers are proxies.** vLLM/cuBLASLt FP8: per-token A scale and per-channel W scale (one fp32 multiply in the
  epilogue). DeepGEMM FP8: UE8M0 power-of-two scale per 1x128 on A and per row-of-128 on B, applied per K-block inside the mainloop (more work, same bytes).
  DeepGEMM FP4 is MXF4 (UE8M0, block 32), not our NVFP4 (E4M3 block 16 + global scale); it cannot load our checkpoint. FP8/FP4 grouped runs use random
  weights/activations (timing only; correctness checked against an fp32 dequantized reference for FP8, calc_diff <= 1.5e-6).
* The DeepGEMM grouped numbers are **GEMM1 + GEMM2 only**: no routing/permute, SwiGLU, re-quantization or finalize. Our column "ours" in the kernel table is the
  two CUTLASS GEMM kernels of the FlashInfer call and the glue is listed separately.
* Routing: layer-19 router recomputed from the captured hidden states on the GPU (bf16 router GEMM, softmax, top-8) = the trace of `kernels/moe-nvfp4`
  (65536 routed rows, 243 active experts, max 6007). Decode uses 3 random token subsets (seeds 0-2) of that trace: 3/12/48/192 tokens x top-8.
  Expert shapes from the model config: hidden 2048, moe_intermediate 512, 256 experts -> GEMM1 (gate+up) N = 1024, K = 2048; GEMM2 (down) N = 2048, K = 512.

## 1. Build and DeepGEMM's own tests

* Build: ~2 min, no problem with nvcc 13.0.88; JIT kernels compile for `sm_120f`.
* `tests/test_sm120_gemm.py`: **73 passed** (647 s). `tests/test_sm120_gemm_regressions.py`: **342 passed** (50 in the first pass, 292 in the second; the
  one remaining item per file, `test_filter`, is the helper imported from `deep_gemm.testing` that pytest mis-collects as a test and is deselected).
  This includes `test_sm120_quant_grouped_graph_mutation`, `test_sm120_dense_fp8_fp4_graph_pdl`, `test_sm120_k_grouped_fp8_graph` and the
  `grouped_tma_independent_streams` tests. Not run: `tests/test_fp8_fp4.py` / `test_bf16.py` (perf scripts), attention/MQA tests.

## 2. Dense FP8 GEMM, our real shapes (mean of two passes, us per call)

`DG` = DeepGEMM `fp8_fp4_gemm_nt` with pre-packed scales (scale packing/quantization not counted for anyone). `DG+PDL` = `set_pdl(True)`.
`our skinny` = the shipped `kernels/dense-fp8-gemm/skinny` kernel (copy built in `ours_ref/`), M <= 64. Tile = BLOCK_M x BLOCK_N x BLOCK_K and pipeline
stages picked by the heuristic, read from the JIT'd kernel's template arguments. `swapAB` = DeepGEMM's native M <= 16 path (the weights become the MMA A operand, so
BLOCK_M/BLOCK_N in the label refer to weight rows / tokens); token count is a compile-time constant of that kernel.

| shape (N x K) | M | DeepGEMM us | DG+PDL us | vLLM CUTLASS us | cuBLASLt us | our skinny us | DG / vLLM | DG TFLOPS | vLLM TFLOPS | DG GB/s | DG tile (BMxBNxBK, stages) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gdn_qkvz (12288x2048) | 1 | 18.6 | 18.5 | 20.9 | 21.5 | 18.1 | 0.89 | 3 | 2 | 1355 | 128x16x64 st10 swapAB |
| gdn_qkvz (12288x2048) | 2 | 18.6 | 18.5 | 21.0 | 21.6 | 18.1 | 0.89 | 5 | 5 | 1354 | 128x16x64 st10 swapAB |
| gdn_qkvz (12288x2048) | 3 | 18.7 | 18.6 | 21.0 | 21.6 | 18.1 | 0.89 | 8 | 7 | 1349 | 128x16x64 st10 swapAB |
| gdn_qkvz (12288x2048) | 4 | 18.7 | 18.6 | 21.0 | 21.5 | 18.1 | 0.89 | 11 | 10 | 1353 | 128x16x64 st10 swapAB |
| gdn_qkvz (12288x2048) | 8 | 18.6 | 18.5 | 21.0 | 21.5 | 18.1 | 0.89 | 22 | 19 | 1362 | 128x16x64 st10 swapAB |
| gdn_qkvz (12288x2048) | 16 | 18.7 | 18.5 | 20.9 | 21.5 | 18.0 | 0.90 | 43 | 39 | 1366 | 128x16x64 st10 swapAB |
| gdn_qkvz (12288x2048) | 64 | 19.1 | 18.9 | 21.8 | 21.0 | 21.2 | 0.87 | 169 | 148 | 1410 | 64x128x128 st3 |
| gdn_qkvz (12288x2048) | 256 | 33.6 | 33.5 | 31.1 | 33.3 | - | 1.08 | 384 | 414 | 953 | 128x128x128 st2 |
| gdn_qkvz (12288x2048) | 1024 | 90.5 | 90.7 | 89.7 | 89.1 | - | 1.01 | 569 | 575 | 579 | 128x128x128 st2 |
| gdn_qkvz (12288x2048) | 8192 | 657.1 | 663.5 | 608.8 | 606.0 | - | 1.08 | 628 | 677 | 370 | 128x128x64 st4 |
| gdn_out (2048x4096) | 1 | 9.7 | 9.3 | 9.4 | 18.3 | 7.6 | 1.03 | 2 | 2 | 866 | 64x16x64 st16 swapAB |
| gdn_out (2048x4096) | 2 | 10.3 | 9.7 | 9.4 | 18.5 | 7.5 | 1.10 | 3 | 4 | 814 | 64x16x64 st16 swapAB |
| gdn_out (2048x4096) | 3 | 10.6 | 9.8 | 9.4 | 18.6 | 7.5 | 1.12 | 5 | 5 | 797 | 64x16x64 st16 swapAB |
| gdn_out (2048x4096) | 4 | 10.5 | 9.9 | 9.5 | 18.6 | 7.6 | 1.11 | 6 | 7 | 799 | 64x16x64 st16 swapAB |
| gdn_out (2048x4096) | 8 | 10.5 | 10.0 | 9.4 | 18.6 | 7.6 | 1.11 | 13 | 14 | 807 | 64x16x64 st16 swapAB |
| gdn_out (2048x4096) | 16 | 10.6 | 10.3 | 9.4 | 18.6 | 7.6 | 1.13 | 25 | 29 | 803 | 64x16x64 st16 swapAB |
| gdn_out (2048x4096) | 64 | 11.6 | 11.4 | 10.8 | 18.7 | 9.1 | 1.07 | 92 | 99 | 767 | 64x64x128 st5 |
| gdn_out (2048x4096) | 256 | 16.9 | 16.7 | 13.0 | 16.7 | - | 1.30 | 254 | 330 | 620 | 128x128x128 st2 |
| gdn_out (2048x4096) | 1024 | 32.5 | 32.3 | 30.1 | 29.9 | - | 1.08 | 529 | 571 | 516 | 128x128x128 st2 |
| gdn_out (2048x4096) | 8192 | 215.8 | 215.1 | 203.0 | 202.1 | - | 1.06 | 637 | 677 | 350 | 128x128x64 st4 |
| attn_qkv (9216x2048) | 1 | 14.8 | 14.7 | 15.6 | 18.4 | 14.2 | 0.95 | 3 | 2 | 1277 | 64x16x64 st16 swapAB |
| attn_qkv (9216x2048) | 2 | 14.8 | 14.6 | 15.5 | 18.3 | 14.3 | 0.96 | 5 | 5 | 1277 | 64x16x64 st16 swapAB |
| attn_qkv (9216x2048) | 3 | 14.8 | 14.7 | 15.3 | 18.0 | 14.0 | 0.97 | 8 | 7 | 1277 | 64x16x64 st16 swapAB |
| attn_qkv (9216x2048) | 4 | 14.8 | 14.7 | 15.5 | 18.3 | 14.2 | 0.96 | 10 | 10 | 1280 | 64x16x64 st16 swapAB |
| attn_qkv (9216x2048) | 8 | 14.8 | 14.7 | 15.4 | 18.1 | 14.2 | 0.96 | 20 | 20 | 1288 | 64x16x64 st16 swapAB |
| attn_qkv (9216x2048) | 16 | 14.9 | 14.7 | 15.5 | 18.2 | 14.2 | 0.96 | 41 | 39 | 1291 | 64x16x64 st16 swapAB |
| attn_qkv (9216x2048) | 64 | 15.2 | 15.2 | 15.8 | 17.7 | - | 0.96 | 159 | 153 | 1329 | 64x64x128 st5 |
| attn_qkv (9216x2048) | 256 | 20.3 | 20.3 | 25.0 | 19.9 | - | 0.81 | 476 | 386 | 1187 | 128x128x128 st2 |
| attn_qkv (9216x2048) | 1024 | 76.4 | 76.6 | 75.2 | 75.0 | - | 1.02 | 506 | 514 | 522 | 128x128x128 st2 |
| attn_qkv (9216x2048) | 8192 | 493.6 | 493.3 | 459.0 | 457.8 | - | 1.08 | 626 | 674 | 378 | 128x128x64 st4 |
| attn_o (2048x4096) | 1 | 10.0 | 9.3 | 9.3 | 18.1 | 7.1 | 1.06 | 2 | 2 | 844 | 64x16x64 st16 swapAB |
| attn_o (2048x4096) | 2 | 10.2 | 9.8 | 9.3 | 18.2 | 7.3 | 1.10 | 3 | 4 | 825 | 64x16x64 st16 swapAB |
| attn_o (2048x4096) | 3 | 10.4 | 9.8 | 9.4 | 18.5 | 7.6 | 1.11 | 5 | 5 | 806 | 64x16x64 st16 swapAB |
| attn_o (2048x4096) | 4 | 10.5 | 9.8 | 9.4 | 18.6 | 7.6 | 1.11 | 6 | 7 | 803 | 64x16x64 st16 swapAB |
| attn_o (2048x4096) | 8 | 10.5 | 9.9 | 9.4 | 18.6 | 7.6 | 1.11 | 13 | 14 | 807 | 64x16x64 st16 swapAB |
| attn_o (2048x4096) | 16 | 10.7 | 10.3 | 9.4 | 18.6 | 7.6 | 1.13 | 25 | 28 | 800 | 64x16x64 st16 swapAB |
| attn_o (2048x4096) | 64 | 11.4 | 11.3 | 10.9 | 18.7 | 9.7 | 1.05 | 94 | 99 | 781 | 64x64x128 st5 |
| attn_o (2048x4096) | 256 | 17.3 | 16.7 | 13.0 | 16.9 | - | 1.32 | 249 | 329 | 607 | 128x128x128 st2 |
| attn_o (2048x4096) | 1024 | 32.6 | 32.5 | 30.3 | 30.1 | - | 1.08 | 526 | 567 | 514 | 128x128x128 st2 |
| attn_o (2048x4096) | 8192 | 215.9 | 215.7 | 203.3 | 202.6 | - | 1.06 | 637 | 676 | 350 | 128x128x64 st4 |
| lm_head (248320x2048) | 1 | 316.9 | 316.7 | 315.8 | 470.9 | 312.9 | 1.00 | 3 | 3 | 1606 | 128x16x64 st10 swapAB |
| lm_head (248320x2048) | 2 | 317.6 | 317.3 | 316.5 | 407.6 | 313.4 | 1.00 | 6 | 6 | 1604 | 128x16x64 st10 swapAB |
| lm_head (248320x2048) | 3 | 317.9 | 317.6 | 317.2 | 474.4 | 314.0 | 1.00 | 10 | 10 | 1605 | 128x16x64 st10 swapAB |
| lm_head (248320x2048) | 4 | 318.7 | 318.3 | 317.5 | 471.2 | 314.6 | 1.00 | 13 | 13 | 1602 | 128x16x64 st10 swapAB |
| lm_head (248320x2048) | 8 | 320.2 | 320.0 | 319.0 | 446.6 | 316.2 | 1.00 | 25 | 26 | 1601 | 128x16x64 st10 swapAB |
| lm_head (248320x2048) | 16 | 322.5 | 322.3 | 321.0 | 424.3 | 318.4 | 1.00 | 50 | 51 | 1602 | 128x16x64 st10 swapAB |
| lm_head (248320x2048) | 64 | 330.6 | 330.6 | 326.4 | 334.8 | - | 1.01 | 197 | 199 | 1635 | 64x128x128 st3 |
| lm_head (248320x2048) | 256 | 441.5 | 442.8 | 578.2 | 436.9 | - | 0.76 | 590 | 450 | 1441 | 128x128x128 st2 |

Also measured (`results/dense3.jsonl`), us DG / vLLM: lm_head M=96 352 / 339, M=128 367 / 356, M=192 **432 / 447**; gdn_qkvz M=192 33.7 / 25.8; attn_qkv M=192 19.6 / 20.3; gdn_out M=192 16.4 / 11.4.

Reading:
* Decode (M <= 16): DeepGEMM beats vLLM's CUTLASS on the two wide-N shapes (gdn_qkvz -11 %, attn_qkv -4 %), ties lm_head, loses ~10 % on the two N = 2048, K = 4096 shapes.
  Our skinny kernel beats DeepGEMM on all four (18.1 vs 18.6, 7.6 vs 10.5, 14.2 vs 14.8, 313 vs 317 us). At M = 32-64 DeepGEMM's 64x128x128 tile is faster than the skinny kernel on
  gdn_qkvz only (19.4 vs 20.1 us at M = 32, 19.4 vs 20.6 at 48, 19.1 vs 21.2 at 64; vLLM 21.0 / 22.3 / 21.8), i.e. ~1.2 us x 30 layers = 0.04 ms per step at c = 16; skinny stays ahead on gdn_out (9.2 vs 10.3 at M = 48) and lm_head (321 vs 330). PDL gives DeepGEMM 0-0.7 us on the small shapes (pdl-swapab covers PDL).
* Prefill (M = 8192): 628 / 637 / 626 / 637 TFLOPS vs vLLM 677 / 677 / 674 / 676: DeepGEMM is 6-8 % slower (tile 128x128x64, 4 stages; its heuristic only
  uses BLOCK_K = 64 above M = 2048). M = 1024: 1-8 % slower. cuBLASLt (rowwise `_scaled_mm`) ties vLLM.
* M = 256: DeepGEMM is 19 % / 24 % faster on attn_qkv / lm_head (vLLM uses its 64x64 tile up to M = 256) and 8 % / 30 % slower on gdn_qkvz / the K = 4096 shapes.
  Not a win overall; the M = 192 case (c = 64) where a 64x128x128 tile beats vLLM's 64x64 is already documented in `kernels/dense-fp8-gemm/README.md`, and DeepGEMM's 432 us there is within 1 % of that note's 431.6 us.

## 3. M-grouped GEMM on the real routing

### 3a. Prefill (contiguous layout, 8192 tokens, layer-19 routing), us per GEMM, cold weights

Contiguous layout needs each expert's rows aligned to BLOCK_M (`set_mk_alignment_for_contiguous_layout`: 32/64/128). `labels` = per-row expert index (`m_indices`),
`psum` = cumulative end offsets (`use_psum_layout`). `loop` = cuBLASLt `_scaled_mm` per active expert (rows padded to 16), the plain baseline.

| routing | variant | GEMM1 us | GEMM2 us | sum us | padded rows | useful TFLOPS g1 / g2 | g1 tile | g2 tile |
|---|---|---|---|---|---|---|---|---|
| L19 | FP8 labels align 128 | 633 | 448 | 1081 | x1.27 | 434 / 307 | 128x128x64 st4 | 128x128x64 st4 |
| L19 | FP8 psum align 128 | 597 | 441 | 1038 | x1.27 | 461 / 311 | 128x128x128 st2 | 128x128x128 st2 |
| L19 | FP8 labels align 64 | 674 | 412 | 1087 | x1.13 | 408 / 333 | 64x128x64 st6 | 64x128x64 st6 |
| L19 | FP8 psum align 64 | 629 | 409 | 1038 | x1.13 | 437 / 336 | 64x128x128 st3 | 64x128x128 st3 |
| L19 | FP8 labels align 32 | 923 | 494 | 1417 | x1.06 | 298 / 278 | 32x64x128 st7 | 32x64x128 st7 |
| L19 | FP8 psum align 32 | 923 | 499 | 1422 | x1.06 | 298 / 275 | 32x64x128 st7 | 32x64x128 st7 |
| L19 | FP4 (MXF4 proxy) labels align 128 | 402 | 471 | 873 | x1.27 | 684 / 292 | 128x128x128 st5 | 128x128x128 st5 |
| L19 | FP4 (MXF4 proxy) psum align 128 | 357 | 364 | 721 | x1.27 | 770 / 378 | 128x128x128 st4 | 128x128x128 st4 |
| L19 | FP4 (MXF4 proxy) psum align 64 | 375 | 329 | 705 | x1.13 | 732 / 417 | 64x128x128 st6 | 64x128x128 st6 |
| L19 | FP8 cuBLASLt loop (per active expert) | 3176 | 2104 | 5280 | x1.03 | 87 / 65 | - | - |
| L0 | FP8 labels align 128 | 636 | 456 | 1091 | x1.28 | 433 / 302 | 128x128x64 st4 | 128x128x64 st4 |
| L0 | FP8 psum align 128 | 597 | 451 | 1047 | x1.28 | 461 / 305 | 128x128x128 st2 | 128x128x128 st2 |
| L0 | FP8 labels align 64 | 675 | 418 | 1093 | x1.13 | 407 / 329 | 64x128x64 st6 | 64x128x64 st6 |
| L0 | FP8 psum align 64 | 649 | 416 | 1065 | x1.13 | 424 / 330 | 64x128x128 st3 | 64x128x128 st3 |
| L0 | FP8 labels align 32 | 924 | 493 | 1418 | x1.06 | 297 / 279 | 32x64x128 st7 | 32x64x128 st7 |
| L0 | FP8 psum align 32 | 923 | 495 | 1417 | x1.06 | 298 / 278 | 32x64x128 st7 | 32x64x128 st7 |
| L0 | FP8 cuBLASLt loop (per active expert) | 3277 | 2158 | 5435 | x1.03 | 84 / 64 | - | - |
| uniform | FP8 labels align 128 | 528 | 393 | 921 | x1.00 | 521 / 350 | 128x128x64 st4 | 128x128x64 st4 |
| uniform | FP8 psum align 128 | 525 | 389 | 914 | x1.00 | 524 / 354 | 128x128x128 st2 | 128x128x128 st2 |
| uniform | FP8 labels align 64 | 596 | 390 | 986 | x1.00 | 461 / 352 | 64x128x64 st6 | 64x128x64 st6 |
| uniform | FP8 psum align 64 | 562 | 389 | 951 | x1.00 | 489 / 353 | 64x128x128 st3 | 64x128x128 st3 |
| uniform | FP8 labels align 32 | 874 | 465 | 1339 | x1.00 | 314 / 296 | 32x64x128 st7 | 32x64x128 st7 |
| uniform | FP8 psum align 32 | 875 | 466 | 1341 | x1.00 | 314 / 295 | 32x64x128 st7 | 32x64x128 st7 |
| uniform | FP4 (MXF4 proxy) labels align 128 | 351 | 434 | 785 | x1.00 | 782 / 317 | 128x128x128 st5 | 128x128x128 st5 |
| uniform | FP4 (MXF4 proxy) psum align 128 | 320 | 304 | 624 | x1.00 | 858 / 453 | 128x128x128 st4 | 128x128x128 st4 |
| uniform | FP4 (MXF4 proxy) psum align 64 | 335 | 305 | 640 | x1.00 | 820 / 451 | 64x128x128 st6 | 64x128x128 st6 |
| uniform | FP8 cuBLASLt loop (per active expert) | 2753 | 1623 | 4377 | x1.00 | 100 / 85 | - | - |

* Hottest expert 6007 rows -> 243 active experts, padding x1.27 at align 128. Uniform routing (256 rows each, no padding) is 12 % faster for the same useful work, consistent with the x1.27 padded rows (+14 % time) rather than with a load imbalance: every tile has the same size and the grid is persistent (188 CTAs, tile index `iter * 188 + blockIdx.x`),
  ~5200 tiles per GEMM1 = 28 waves, so the tail is negligible. There is no scheduling feature for skew in this scheduler.
* Alignment: 128 and 64 tie on FP8 (1038 us each with psum); 64 is 2-3 % better for FP4 psum (705 vs 721 us in graph timing); 32 (BLOCK_M = 32) is 37 % slower at this size.
* `psum` beats `labels`: -6 % on FP8 GEMM1 (no change on GEMM2); FP4 at align 128: -11 % GEMM1, -23 % GEMM2. Our FlashInfer path already consumes per-expert offsets, so there is nothing to transfer.
* Against our actual path (kernel device time, cold weights, same routing; glue is `expandInputRows` 140 us + `doActivation` 96 us + prefix sums/strides ~37 us, nothing like it is in the DeepGEMM numbers):

| T tokens | ours GEMM1 / GEMM2 us (NVFP4 CUTLASS) | ours glue us | DG FP4 (MXF4) labels BM32 GEMM1 / GEMM2 us | DG vs ours (GEMMs) | DG FP8 labels BM32 g1 / g2 us |
|---|---|---|---|---|---|
| 3 | 20.5 / 13.0 = 33.5 | 11.3 | 12.9 / 5.6 = 18.5 | -45% | 29.3 / 10.1 |
| 12 | 51.3 / 28.6 = 80.0 | 11.9 | 46.3 / 24.4 = 70.7 | -12% | 86.8 / 43.2 |
| 48 | 112.9 / 64.7 = 177.6 | 15.6 | 105.6 / 53.2 = 158.8 | -11% | 204.0 / 101.9 |
| 192 | 150.4 / 81.4 = 231.8 | 20.4 | 140.9 / 70.4 = 211.3 | -9% | 269.7 / 134.6 |
| 8192 (L19) | 334.5 / 240.9 = 575.4 | 281.8 | psum al128: 353.0 / 358.6 = 711.6 | +24% | psum al128: 579.3 / 437.9 |
| 8192 (L19) | 334.5 / 240.9 = 575.4 | 281.8 | psum al64: 365.2 / 324.3 = 689.5 | +20% | psum al128: 579.3 / 437.9 |
| 8192 (L19) | 334.5 / 240.9 = 575.4 | 281.8 | labels al128: 385.7 / 463.0 = 848.7 | +47% | psum al128: 579.3 / 437.9 |

  DeepGEMM's FP4 GEMM1 is 6 % slower than ours, GEMM2 (K = 512, no fused finalize) 35-49 % slower. FP8 costs 1.4x (prefill) to 1.9x (decode) the FP4 time.
* The loop baseline is 5x slower than DeepGEMM (5.3 ms vs 1.04 ms for FP8 L19; 87 vs 461 useful TFLOPS on GEMM1): 243 small per-expert GEMMs.

### 3b. Decode (3, 12, 48, 192 tokens x top-8; mean of 3 token subsets), us, GEMM1 + GEMM2

`masked` = per-expert counts with a fixed [G, max_m, K] layout (the no-sync layout for CUDA graphs); contiguous with align 32 selects BLOCK_M = 32 (tile 32x64x128, 7 stages FP8 / 15 FP4).
Weight TB/s counts weight bytes of the active experts only.

| T tokens (x top-8) | active experts (seed 0/1/2) | variant | GEMM1 us | GEMM2 us | sum us | weight TB/s (g1) |
|---|---|---|---|---|---|---|
| 3 | 21/23/21 | FP8 contiguous labels, align 32 (BM=32) | 32.2 | 12.9 | 45.2 | 1.41 |
| 3 | 21/23/21 | FP8 contiguous psum, align 32 (BM=32) | 35.3 | 22.0 | 57.2 | 1.29 |
| 3 | 21/23/21 | FP8 loop of cuBLASLt _scaled_mm per active expert | 262.5 | 164.4 | 426.9 | 0.17 |
| 3 | 21/23/21 | FP4 (MXF4) contiguous labels, align 32 (BM=32) | 13.7 | 7.5 | 21.2 | 1.61 |
| 3 | 21/23/21 | FP4 (MXF4) contiguous psum, align 32 (BM=32) | 23.5 | 20.1 | 43.7 | 0.94 |
| 3 | 21/23/21 | FP8 masked, align 32 (BM=32) | 44.1 | 37.4 | 81.5 | 1.03 |
| 3 | 21/23/21 | FP8 masked, align 64 | 50.4 | 39.2 | 89.6 | 0.90 |
| 3 | 21/23/21 | FP8 masked, align 128 | 50.4 | 39.2 | 89.6 | 0.90 |
| 3 | 21/23/21 | FP4 masked, align 32 (BM=32) | 23.9 | 21.4 | 45.3 | 0.92 |
| 3 | 21/23/21 | FP4 masked, align 64 | 24.5 | 21.5 | 46.0 | 0.90 |
| 3 | 21/23/21 | FP4 masked, align 128 | 24.4 | 21.7 | 46.1 | 0.90 |
| 12 | 64/60/65 | FP8 contiguous labels, align 32 (BM=32) | 87.7 | 44.9 | 132.6 | 1.51 |
| 12 | 64/60/65 | FP8 contiguous psum, align 32 (BM=32) | 89.0 | 45.6 | 134.6 | 1.48 |
| 12 | 64/60/65 | FP8 loop of cuBLASLt _scaled_mm per active expert | 783.6 | 472.7 | 1256.3 | 0.17 |
| 12 | 64/60/65 | FP4 (MXF4) contiguous labels, align 32 (BM=32) | 48.1 | 26.3 | 74.4 | 1.40 |
| 12 | 64/60/65 | FP4 (MXF4) contiguous psum, align 32 (BM=32) | 48.5 | 29.3 | 77.8 | 1.38 |
| 12 | 64/60/65 | FP8 masked, align 32 (BM=32) | 97.1 | 59.2 | 156.3 | 1.36 |
| 12 | 64/60/65 | FP8 masked, align 64 | 103.3 | 62.3 | 165.6 | 1.28 |
| 12 | 64/60/65 | FP8 masked, align 128 | 103.3 | 62.4 | 165.7 | 1.28 |
| 12 | 64/60/65 | FP4 masked, align 32 (BM=32) | 48.6 | 30.0 | 78.5 | 1.38 |
| 12 | 64/60/65 | FP4 masked, align 64 | 50.3 | 30.0 | 80.3 | 1.33 |
| 12 | 64/60/65 | FP4 masked, align 128 | 50.3 | 29.8 | 80.1 | 1.33 |
| 48 | 147/137/128 | FP8 contiguous labels, align 32 (BM=32) | 192.5 | 97.8 | 290.2 | 1.50 |
| 48 | 147/137/128 | FP8 contiguous psum, align 32 (BM=32) | 192.3 | 98.4 | 290.8 | 1.50 |
| 48 | 147/137/128 | FP8 loop of cuBLASLt _scaled_mm per active expert | 1713.3 | 1119.6 | 2832.9 | 0.17 |
| 48 | 147/137/128 | FP4 (MXF4) contiguous labels, align 32 (BM=32) | 107.6 | 55.1 | 162.8 | 1.43 |
| 48 | 147/137/128 | FP4 (MXF4) contiguous psum, align 32 (BM=32) | 112.9 | 55.5 | 168.4 | 1.36 |
| 48 | 147/137/128 | FP8 masked, align 32 (BM=32) | 219.1 | 147.9 | 367.0 | 1.31 |
| 48 | 147/137/128 | FP8 masked, align 64 | 226.2 | 154.0 | 380.2 | 1.27 |
| 48 | 147/137/128 | FP8 masked, align 128 | 226.4 | 154.0 | 380.4 | 1.27 |
| 48 | 147/137/128 | FP4 masked, align 32 (BM=32) | 108.5 | 55.1 | 163.6 | 1.42 |
| 48 | 147/137/128 | FP4 masked, align 64 | 112.2 | 57.8 | 169.9 | 1.37 |
| 48 | 147/137/128 | FP4 masked, align 128 | 112.1 | 57.9 | 170.0 | 1.38 |
| 192 | 194/200/198 | FP8 contiguous labels, align 32 (BM=32) | 275.5 | 139.6 | 415.1 | 1.50 |
| 192 | 194/200/198 | FP8 contiguous psum, align 32 (BM=32) | 275.1 | 140.0 | 415.0 | 1.50 |
| 192 | 194/200/198 | FP8 loop of cuBLASLt _scaled_mm per active expert | 2513.3 | 1641.1 | 4154.4 | 0.16 |
| 192 | 194/200/198 | FP4 (MXF4) contiguous labels, align 32 (BM=32) | 143.2 | 72.6 | 215.8 | 1.42 |
| 192 | 194/200/198 | FP4 (MXF4) contiguous psum, align 32 (BM=32) | 147.7 | 75.9 | 223.7 | 1.38 |
| 192 | 194/200/198 | FP8 masked, align 32 (BM=32) | 274.3 | 139.0 | 413.2 | 1.51 |
| 192 | 194/200/198 | FP8 masked, align 64 | 284.7 | 168.2 | 452.9 | 1.45 |
| 192 | 194/200/198 | FP8 masked, align 128 | 284.8 | 162.4 | 447.2 | 1.45 |
| 192 | 194/200/198 | FP4 masked, align 32 (BM=32) | 143.3 | 72.3 | 215.6 | 1.42 |
| 192 | 194/200/198 | FP4 masked, align 64 | 149.8 | 78.5 | 228.2 | 1.36 |
| 192 | 194/200/198 | FP4 masked, align 128 | 149.6 | 78.4 | 228.0 | 1.36 |

Our whole FlashInfer call (including glue), warm = timed like `bench_tactics.py` (same weights every call), cold = 6 rotated copies:

| T | active experts | ours warm (bench_tactics style) | ours cold (6 weight copies) | DG FP4 masked BM32 | DG FP4 contiguous labels BM32 | DG FP4 contiguous psum BM32 | DG FP8 contiguous labels BM32 | DG FP8 masked BM32 |
|---|---|---|---|---|---|---|---|---|
| 3 | 21/23/21 | 33.4 | 44.4 | 45.3 | 21.2 | 43.7 | 45.2 | 81.5 |
| 12 | 64/60/65 | 51.5 | 92.1 | 78.5 | 74.4 | 77.8 | 132.6 | 156.3 |
| 48 | 147/137/128 | 182.7 | 182.4 | 163.6 | 162.8 | 168.4 | 290.2 | 367.0 |
| 192 | 194/200/198 | 257.3 | 256.8 | 215.6 | 215.8 | 223.7 | 415.1 | 413.2 |
| 8192 | 243 | 852.6 | 858.5 | n/a | FP4 labels al128: 872.6 / psum al128: 720.7 / psum al64: 704.6 | | FP8 psum al128: 1037.8 | |

* **Timing caveat for the existing `kernels/moe-nvfp4` decode numbers:** reusing the same weights lets the 128 MB L2 serve the small batches. Cold, the same call is 1.3x slower at 3 tokens (44 vs 33 us)
  and 1.8x at 12 tokens (92 vs 51 us); from 48 tokens on it is the same (the working set no longer fits).
* DeepGEMM's masked layout is slower than its contiguous layout at small T (FP8 81 vs 45 us at 3 tokens; 156 vs 133 at 12). Contiguous BLOCK_M = 32 reaches 1.5 TB/s of FP8 weight traffic (83 % of the 1.8 TB/s ideal).
* Forcing the BM=32 kernel to BLOCK_N = 128 instead of 64 (patched copy) to test whether tile granularity explains the gap to our CUTLASS GEMMs:

| T | BLOCK_N=64 (upstream) g1+g2 us | BLOCK_N=128 g1+g2 us |
|---|---|---|
| 3 | 18.5 | 16.8 |
| 12 | 70.7 | 71.1 |
| 48 | 158.8 | 159.1 |
| 192 | 211.3 | 212.8 |

  No effect, so the DeepGEMM-vs-CUTLASS decode gap is not a tile-size/scheduling-granularity effect I could identify. Candidates I did not test: the grouped CUTLASS kernel scanning 256 group problems
  when only 21 are non-empty, pipeline depth (DeepGEMM uses 15 stages at BM=32), different TMA/scale-factor loading.

## 4. CUDA-graph safety (issue #444)

Protocol (`graph_safety.py`): static A / SFA / B / SFB / D buffers and a static routing tensor. Capture once with routing A (real L19), then 1000 replays per
configuration; before each replay the routing tensor is overwritten in place (same pointer) with one of 48 routings: real L19, uniform, 8 hot experts, 2 experts, even experts only,
3 rows total, all empty, decode-like, 40 random Dirichlet skews; D is pre-filled with NaN; after each replay `synchronize()` (an illegal access would surface here) and a bitwise compare of all valid
rows with an eager call on the same inputs; every 50th replay also an fp32 reference of 3 experts. Psum mode re-packs SFA for the current routing before every replay, as a real quant kernel would.

| mode | experts G | N x K | replays | bitwise mismatch vs eager | fp32-ref mismatch | error |
|---|---|---|---|---|---|---|
| fp8_labels128 | 256 | 1024x2048 | 1000 | 0 | 0 | None |
| fp8_labels32 | 256 | 1024x2048 | 1000 | 0 | 0 | None |
| fp8_psum128 | 256 | 1024x2048 | 1000 | 0 | 0 | None |
| fp8_masked | 256 | 1024x2048 | 1000 | 0 | 0 | None |
| fp8_labels128_zp | 256 | 1024x2048 | 1000 | 0 | 0 | None |
| fp8_labels128_pdl | 256 | 1024x2048 | 1000 | 0 | 0 | None |
| fp4_labels128 | 256 | 1024x2048 | 1000 | 0 | 0 | None |
| fp8_labels128_G288 | 288 | 512x4096 | 1000 | 0 | 0 | None |
| fp8_labels128_G288_w2 | 288 | 4096x256 | 1000 | 0 | 0 | None |
| fp8_masked_G288 | 288 | 512x4096 | 1000 | 0 | 0 | None |
| fp8_masked_G288_w2 | 288 | 4096x256 | 1000 | 0 | 0 | None |
| fp8_psum128_G288 | 288 | 512x4096 | 1000 | 0 | 0 | None |
| fp8_masked_w2 | 256 | 2048x512 | 1000 | 0 | 0 | None |
| fp8_labels128_w2 | 256 | 2048x512 | 1000 | 0 | 0 | None |

* Result: **#444 does not reproduce**; 14 000 replays, zero errors, zero mismatches, including `ensure_zero_padding=True` (the API default, adds a clear-padding kernel), PDL on, FP4, and the issue's
  own shapes (G = 288, N/K = 512/4096 and 4096/256) in contiguous, psum and masked layouts.
* Why, from reading the code: the SM120 m-grouped kernels receive A/B/SF/CD tensor maps as host-built `__grid_constant__` kernel parameters; routing (`m_indices`, psum, `masked_m`) is re-read from global
  memory on every launch. The only on-device `tensor_map_replace_*` (the mechanism suspected in the issue) is in the k-grouped path (`is_k_grouped_contiguous`), also in the older commit 33a715e.
* Limits: the reporter also saw a clean synthetic chain on the affected wheel and a crash only with the dumped serving arguments of their sglang run, so a clean synthetic run cannot prove their
  crash is gone; I did not run their wheel (0.1.5.post3) or the pre-#447 tree, did not run `compute-sanitizer`, and did not capture a graph with sglang's exact call sequence.

## 5. Conclusion

Is there a mechanism that is measurably better than ours **and** applicable to our NVFP4 kernels without a new quantization contract? **No.**

* Dense: nothing (ours is faster at decode, vLLM/CUTLASS is faster at prefill).
* Prefill grouped: DeepGEMM is slower than our CUTLASS GEMMs even in the friendlier MXF4 format; its scheduler has no skew mechanism (padding, not imbalance, is the cost of skew; our CUTLASS grouped GEMM takes exact per-expert problem sizes and has no explicit row padding).
* Layout ideas (psum offsets, BLOCK_M chosen from expected rows, masked layout): we already consume offsets; BLOCK_M = 32 is what our 128x32 swapAB tactic is; masked is slower than contiguous in DeepGEMM too.
* The only measured difference is the decode GEMM kernel speed (up to 45 % at 3 tokens, 9-12 % at 12-192 tokens, of which ~6 points are the smaller scale format). That is a kernel-implementation difference I could not
  attribute to a transferable mechanism, and exploiting it would mean writing a new NVFP4 (block-16 E4M3 scale) kernel, i.e. a port. If this is pursued at all, the first step is an ncu comparison of
  the two kernels at 3-12 tokens, not a port.
* Not measured: any NVFP4 kernel from DeepGEMM; sustained power-limited clocks for DeepGEMM; end-to-end vLLM; ncu counters; the pre-#447 tree; BF16 GEMMs and k-grouped (backward) GEMMs; Mega MoE/Mega Gate/Mega mHC (not available on SM120).

## Files

* `setup.sh`, `dg_docker.sh`, `run_gpu.sh`, `jobs/*.sh`: build env and the exact GPU jobs (all run under the shared GPU lock).
* `bench_dense.py`, `bench_grouped.py`, `graph_safety.py`, `bench_ours_cold.py`, `bench_ours_kernels.py`, `bench_dg_kernels.py`, `common.py`, `summarize.py`, `build_readme.py`.
* `results/*.jsonl` (raw), `results/*.log` (job logs). `ours_ref/` = copy of the shipped skinny kernel used as the dense reference. `patches/force_bm32_block_n.patch`.
* `src/` (357 MB DeepGEMM tree + submodules), `home/` (JIT cache) are git-ignored and regenerable with `setup.sh`.

The first actual vLLM integration control (`integration/results/dg_serving_off1`) passed 30-module runtime eligibility auditing and an NSYS gate proving live M48 graph replay with zero candidate kernels. The matched candidate `dg_serving_on1` failed before serving because unit-scale initialization inherited vLLM's BF16 default dtype. Both missing float32 annotations are now fixed and covered by a CPU regression of the actual preparation function. An additional audit found vLLM enabled DeepGEMM PDL globally while the operator microtests used DeepJIT's false default; the controlled plugin now pins false in both profiles and asserts it at startup/shutdown. A fresh same-source pair is required. No end-to-end speedup or model-quality conclusion is available from this first attempt.

Corrected actual-vLLM pair `dg_serving_off2`/`dg_serving_on2` proves all 30 QKVZ replacements execute inside live M48 graphs: 30 fused nodes/replay, median summed candidate time 0.584697 ms versus OFF inferred-QKVZ geometry 0.665625 ms (~12.16% lower). **This is not a selected serving improvement.** One short decode screen gives +0.68% throughput but worse mean TPOT (+4.43%); prefill fallback is flat. NLL differs, three mapped common-source Inductor configs differ, and ON lacks worker shutdown checkpoints (exit5). Keep default-off; full context and immutable evidence are in `integration/README.md` and `integration/results/dg_serving_pair2_summary.json`. Agentic remains reserved for the final selected composition.
