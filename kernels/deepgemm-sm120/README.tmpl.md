# DeepGEMM `nv_dev` SM120 kernels vs our stack (reference study, nothing integrated)

Question: can DeepGEMM's native SM120 kernels (branch `origin/nv_dev`, commit b64107f) or their scheduling ideas beat what
vLLM v0.32.0 runs for `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` on one RTX PRO 6000 Blackwell (SM120, 188 SMs), and are they CUDA-graph safe?
Everything here is a benchmark; the original DeepGEMM clone and vLLM were not touched. Nothing is committed.

## Short answer

* **No kernel or idea from DeepGEMM is worth integrating.** Dense FP8: our shipped skinny kernel (`kernels/dense-fp8-gemm`) is faster
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
* Recommendation: do not port. If anything is followed up, it is a profile (ncu) of our FlashInfer GEMM1 kernel at 3-12 tokens against
  DeepGEMM's BM=32 kernel to find where the ~7 us per GEMM at 3 tokens goes. Upper bound if all of it were recoverable: 15 us x 40 layers = 0.6 ms of a 6.5 ms c=1 step; not demonstrated.

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

{{DENSE}}

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

{{PREFILL}}

* Hottest expert 6007 rows -> 243 active experts, padding x1.27 at align 128. Uniform routing (256 rows each, no padding) is 12 % faster for the same useful work, consistent with the x1.27 padded rows (+14 % time) rather than with a load imbalance: every tile has the same size and the grid is persistent (188 CTAs, tile index `iter * 188 + blockIdx.x`),
  ~5200 tiles per GEMM1 = 28 waves, so the tail is negligible. There is no scheduling feature for skew in this scheduler.
* Alignment: 128 and 64 tie on FP8 (1038 us each with psum); 64 is 2-3 % better for FP4 psum (705 vs 721 us in graph timing); 32 (BLOCK_M = 32) is 37 % slower at this size.
* `psum` beats `labels`: -6 % on FP8 GEMM1 (no change on GEMM2); FP4 at align 128: -11 % GEMM1, -23 % GEMM2. Our FlashInfer path already consumes per-expert offsets, so there is nothing to transfer.
* Against our actual path (kernel device time, cold weights, same routing; glue is `expandInputRows` 140 us + `doActivation` 96 us + prefix sums/strides ~37 us, nothing like it is in the DeepGEMM numbers):

{{KERNEL}}

  DeepGEMM's FP4 GEMM1 is 6 % slower than ours, GEMM2 (K = 512, no fused finalize) 35-49 % slower. FP8 costs 1.4x (prefill) to 1.9x (decode) the FP4 time.
* The loop baseline is 5x slower than DeepGEMM (5.3 ms vs 1.04 ms for FP8 L19; 87 vs 461 useful TFLOPS on GEMM1): 243 small per-expert GEMMs.

### 3b. Decode (3, 12, 48, 192 tokens x top-8; mean of 3 token subsets), us, GEMM1 + GEMM2

`masked` = per-expert counts with a fixed [G, max_m, K] layout (the no-sync layout for CUDA graphs); contiguous with align 32 selects BLOCK_M = 32 (tile 32x64x128, 7 stages FP8 / 15 FP4).
Weight TB/s counts weight bytes of the active experts only.

{{DECODE}}

Our whole FlashInfer call (including glue), warm = timed like `bench_tactics.py` (same weights every call), cold = 6 rotated copies:

{{OURS}}

* **Timing caveat for the existing `kernels/moe-nvfp4` decode numbers:** reusing the same weights lets the 128 MB L2 serve the small batches. Cold, the same call is 1.3x slower at 3 tokens (44 vs 33 us)
  and 1.8x at 12 tokens (92 vs 51 us); from 48 tokens on it is the same (the working set no longer fits).
* DeepGEMM's masked layout is slower than its contiguous layout at small T (FP8 81 vs 45 us at 3 tokens; 156 vs 133 at 12). Contiguous BLOCK_M = 32 reaches 1.5 TB/s of FP8 weight traffic (83 % of the 1.8 TB/s ideal).
* Forcing the BM=32 kernel to BLOCK_N = 128 instead of 64 (patched copy) to test whether tile granularity explains the gap to our CUTLASS GEMMs:

{{BN}}

  No effect, so the DeepGEMM-vs-CUTLASS decode gap is not a tile-size/scheduling-granularity effect I could identify. Candidates I did not test: the grouped CUTLASS kernel scanning 256 group problems
  when only 21 are non-empty, pipeline depth (DeepGEMM uses 15 stages at BM=32), different TMA/scale-factor loading.

## 4. CUDA-graph safety (issue #444)

Protocol (`graph_safety.py`): static A / SFA / B / SFB / D buffers and a static routing tensor. Capture once with routing A (real L19), then 1000 replays per
configuration; before each replay the routing tensor is overwritten in place (same pointer) with one of 48 routings: real L19, uniform, 8 hot experts, 2 experts, even experts only,
3 rows total, all empty, decode-like, 40 random Dirichlet skews; D is pre-filled with NaN; after each replay `synchronize()` (an illegal access would surface here) and a bitwise compare of all valid
rows with an eager call on the same inputs; every 50th replay also an fp32 reference of 3 experts. Psum mode re-packs SFA for the current routing before every replay, as a real quant kernel would.

{{GRAPH}}

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
