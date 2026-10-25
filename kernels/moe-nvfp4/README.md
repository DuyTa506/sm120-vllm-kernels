# MoE (NVFP4 experts) and router on SM120

Area: `kernels/moe-nvfp4/`. Model: unsloth/Qwen3.6-35B-A3B-NVFP4-Fast (256 experts, top-8, hidden 2048,
moe_intermediate 512; compressed-tensors NVFP4 W4A4 with calibrated static activation global scales).
Stack: vLLM v0.32.0, FlashInfer 0.7.0.post1, RTX PRO 6000 Blackwell (SM120).

## Summary

1. **Why b12x lost quality: a scale bug in vLLM's b12x integration.** `FlashInferB12xExperts.process_weights_after_loading`
   multiplies every e4m3 weight block scale by the per-expert weight global scale (`1/weight_global_scale`, about 3e-5)
   and stores the result back as e4m3. Every product lands in the e4m3 subnormal range (below 2^-6). The effective
   block scales end up with a 13-47 % mean relative error, and in layer 0 gate/up 36 % of the blocks are flushed to zero.
   On real weights each expert matmul has a 13-20 % relative error from this alone. The fused-MoE output error against
   an fp32 reference doubles: 19-29 % instead of 7.5-14 % for CUTLASS. Prompt NLL on licenses went from 0.094 to 0.177.
2. **A second b12x issue, inside FlashInfer.** The b12x MMA kernels (micro/static/dynamic, 4 or more tokens) read
   `input_global_scale` and `fc2_input_scale` as reciprocal (dequant-style) scales. The direct-micro kernel (3 or fewer
   tokens) reads them as multipliers. With calibrated scales, one of the two paths is always wrong: either the output
   is exactly zero for 1-3 tokens, or it is off by 1/c^2 for 4 or more tokens. vLLM forces all activation scales to
   1.0, where both readings agree. That is probably why it baked the weight scale into the block scales in the first place.
3. **Fix plus a hybrid backend** (`03-moe-b12x-scale-fix-and-hybrid.patch`, opt-in with `VLLM_B12X_FIX_SCALES=1` and
   `--moe-backend flashinfer_b12x`):
   - The checkpoint block scales are kept unchanged, and the weight global scale is applied in fp32 through the alphas.
   - The calibrated static activation scales are passed in the form the MMA kernels expect.
   - Batches with fewer than 1024 tokens (decode, MTP verify, small chunks) run the existing FlashInfer CUTLASS path,
     with identical weights, scales and autotuned tactics. Larger batches run b12x's fused dynamic kernel.

   At kernel level, the fixed b12x matches CUTLASS's error against fp32 on all 5 tested layers (0.0753 vs 0.0753 …
   0.1380 vs 0.1379). It differs from CUTLASS by 1.6-2.7 % rel-L2, which is FP4 rounding-boundary noise, not bf16 noise
   (flagged). It is **28 % faster than CUTLASS at 8192 tokens** (631 vs 871 us per layer, CUDA graph, layer 19).
   End to end, 8k-prefill TTFT drops from 194.6 to 182.8-188.2 ms (two runs; mean -9 ms, -4.7 %).
   Prompt NLL, needles, MTP acceptance and decode ITL are unchanged within run-to-run noise (section 7).
4. **Tactics:** vLLM autotunes FlashInfer's CUTLASS MoE GEMMs at warmup for every token bucket. A full sweep of the
   SM120 tactic space with real routing found nothing better than the autotuner's pick (within ±1.3 % at M = 3 … 8192).
   There is no gain to take there.
5. **Router GEMM (bf16 2048→256, cuBLAS):** 33 us at 8192 tokens plus 12 us for the shared-expert gate GEMV. The
   bandwidth/compute ideal is about 19 us, so the most this could save is about 1 ms per 8k prefill. Fusing the
   shared-expert gate into the router GEMM saves 9 us per layer (0.36 ms per 8k). I did not pursue it.

## 1. How vLLM picks the MoE path on SM120

- `oracle/nvfp4.py:select_nvfp4_moe_backend` tries, in order, FLASHINFER_TRTLLM, FLASHINFER_CUTEDSL(_BATCHED),
  FLASHINFER_CUTLASS, and so on. On SM120 the first two are unsupported, so **FLASHINFER_CUTLASS** (`FlashInferExperts`) is
  used. That is TRT-LLM's CUTLASS MoE runner as JIT-built by FlashInfer (`fused_moe_120.so` from flashinfer-jit-cache).
  FLASHINFER_B12X is never auto-selected; you have to pass `--moe-backend flashinfer_b12x`.
- Each forward runs:
  1. The router: `GateLinear`, which on SM120 falls back to `F.linear`/cuBLAS bf16. The sm80-named kernels are normal,
     because SM120 has only mma.sync-class tensor cores.
  2. `topkGating` (softmax, top-8, renorm).
  3. The vLLM `scaled_fp4_quant`, using the calibrated global scale `a1_gscale` = min over experts of `input_global_scale`.
  4. FlashInfer `cutlass_fused_moe`, which runs: `blockExpertPrefixSum` → `globalExpertPrefixSum` →
     `mergeExpertPrefixSum` → `expandInputRows` (permute FP4 rows and scale factors) → grouped GEMM1 (bf16 out) →
     `doActivation` (SwiGLU plus FP4 quant with `a2_gscale`) → grouped GEMM2 with fused finalize (weighted scatter-add).
- **Autotuning:** `kernel_warmup.flashinfer_autotune` runs at the default -O2 level. It does one dummy run at
  max_num_batched_tokens inside `autotune(tuning_buckets=…)`, which profiles every token bucket (1 … 16384) for
  `trtllm::fused_moe::gemm1` and `gemm2`. The results are saved in
  `~/.cache/vllm/flashinfer_autotune_cache/<ver>/<arch>/<hash>/autotune_configs_dp0_rank0.json`.
- **Tactic space** (FlashInfer 0.7.0.post1 `get_candidate_configs_sm120` × swap_ab × finalize):
  - GEMM1 has ids 0-19: 10 CTA tiles {128x128x128B, 128x128x64B, 256x128x64B, 128x256x64B, 128x128x256B, 256x128x128B,
    128x32x128B, 128x32x64B, 128x64x128B, 128x64x64B}, each with or without swapAB. K is in bytes, so 64B is 128 FP4
    elements. The occupancy filter drops the tiles that SM120 cannot launch.
  - GEMM2 has ids 20-59: the same tiles, each with or without fused finalize, and with or without swapAB.
  - Autotuned picks: 8192 tokens → GEMM1 12 (256x128x64B+swapAB) and GEMM2 52 (256x128x64B+finalize+swapAB).
    Decode buckets → GEMM1 16/17 (128x32+swapAB) and GEMM2 56/57 (128x32+finalize+swapAB).

## 2. Where the time goes (FlashInfer CUTLASS path, layer 19, real captured inputs and routing)

`bench_breakdown.py` (torch.profiler/CUPTI, eager, median of 20 calls), `results/breakdown_L19.txt`:

| kernel (us/layer)       | M=8192 | M=4096 | M=48 (c=16 decode) | M=12 (c=4) |
|-------------------------|-------:|-------:|-----:|-----:|
| grouped GEMM1 (gate/up) | 334.8 | 247.8 | 42.9 | 16.8 |
| grouped GEMM2 (+finalize)| 238.4 | 159.2 | 26.7 | 12.9 |
| expandInputRows         | 137.9 | 68.9 | 3.2 | 2.7 |
| doActivation            | 95.5 | 41.3 | 3.0 | 2.7 |
| blockExpertPrefixSum    | 41.7 | 14.5 | (buildExpertMaps 3.1) | 3.2 |
| scaled_fp4_quant (vLLM) | 23.1 | 12.4 | 1.5 | 1.0 |
| other (strides, merge, memset) | 23.6 | 15.4 | 2.7 | 2.7 |
| **total**               | **895** | **560** | **83** | **42** |

- The grouped GEMMs reach 719 TFLOPS at 8192 tokens (33 % of the measured 2.2 PFLOPS NVFP4 peak) and 507 TFLOPS at
  4096 tokens.
- The real routing is very skewed. At 8192 tokens, layer 19 has 256 rows per expert on average, but one expert
  receives 6007 of the 8192 tokens, and only 243 experts are active.
- Decode sizes are weight-bandwidth bound. 48 tokens touch 131 experts, about 205 MB of FP4 weights.
- Of the 895 us at 8k, 322 us is glue around the GEMMs (expand, activation, prefix sums, quant).
- `expandInputRows` copies only about 34 MB, yet takes 138 us (about 250 GB/s). It uses one CTA per row and 32-bit loads.
- Fixing the glue inside the CUTLASS path means rebuilding FlashInfer's `fused_moe_120` module, which I did not do here.
  b12x's fused kernel removes this glue anyway.

## 3. Tactic sweep: no gain

`bench_tactics.py` (whole fused-MoE call, CUDA graph, median of 30 replays × 5 calls). Each GEMM1 tactic was timed with
GEMM2 at its default, then each GEMM2 tactic with the best GEMM1. Results in `results/tactics_sweep_L19.txt`.

| M (tokens) | autotuned (us) | best pair found (us) | best vs autotuned |
|---:|---:|---:|---:|
| 8192 | 867.5 | 872.4 [12,52] | -0.6 % (autotuned is better) |
| 1024 | 320.4 | 324.2 [1,49] | -1.2 % |
| 192 | 262.8 | 259.4 [17,47] | +1.3 % |
| 48 | 174.6 | 173.5 [16,46] | +0.6 % |
| 3 | 32.8 | 32.5 [17,57] | +0.8 % |

The autotuner profiles with synthetic balanced routing, yet its picks are within noise of the best with real, skewed
routing. The tile shapes in this space are not the bottleneck.

## 4. b12x numerics

### 4a. Weight-scale bake (vLLM bug)

`check_b12x_scale_bake.py` (CPU, real checkpoint) checks the effective dequant block scale `s/gs` when it is stored as
e4m3, as vLLM does:

```
layer  0 gate_proj: block-scale rel err mean 0.474 p50 0.271 p99 1.000 | subnormal 100.0%  flushed-to-0 36.41%
layer 20 gate_proj: block-scale rel err mean 0.145 p50 0.129 p99 0.327 | subnormal 100.0%  flushed-to-0 0.10%
layer 39 down_proj: block-scale rel err mean 0.131 p50 0.117 p99 0.327 | subnormal 100.0%  flushed-to-0 0.00%
```

`check_b12x_bake_matmul_cpu.py` gives the relative L2 error of W·x with the baked scales versus the exact NVFP4 weights:
0.19 (layer 0), 0.16 (layer 19) and 0.13 (layer 39) for gate, up and down alike.

### 4b. Fused-MoE output vs fp32 reference (GPU, real inputs and routing, 4096 tokens)

`numerics_b12x.py` → `results/numerics_b12x.txt`. The reference uses exactly dequantized weights and unquantized
activations, in fp32.

| layer | CUTLASS vs ref | b12x as in vLLM | b12x fixed (static scales) | fixed vs CUTLASS | b12x unit act. scales | unit vs CUTLASS |
|---:|---:|---:|---:|---:|---:|---:|
| 0  | 0.0753 | 0.1937 | 0.0753 | 0.0163 | 0.0760 | 0.0741 |
| 9  | 0.1264 | 0.2583 | 0.1266 | 0.0236 | 0.1267 | 0.1289 |
| 19 | 0.1360 | 0.2865 | 0.1362 | 0.0248 | 0.1360 | 0.1410 |
| 29 | 0.1379 | 0.2741 | 0.1380 | 0.0268 | 0.1379 | 0.1428 |
| 39 | 0.0997 | 0.1849 | 0.0997 | 0.0216 | 0.0995 | 0.1026 |

All values are rel-L2. "Unit act. scales" means unbaked weights with activation global scales of 1.0, i.e. pure dynamic
per-16 scaling.
- Both fixes restore CUTLASS-level accuracy against fp32.
- With static scales, b12x stays within 1.6-2.7 % of CUTLASS's output. With unit scales the gap is 7-14 %, from a
  different FP4 grid.
- **Flag:** neither is bit-close to CUTLASS. The residual is FP4 rounding-boundary flips, caused by a different
  accumulation and activation path, not bf16 noise.

### 4c. Scale semantics in FlashInfer b12x (FlashInfer issue)

- Probes in `numerics_b12x.py`: passing a quantization multiplier `c` as `fc2_input_scale`, with the matching
  `w2_alpha = 1/(wgs2*c)`, shrinks the output by exactly 1/c^2 (rel-L2 0.754 for c=2 and 0.985 for c=8, with cos 0.99).
  So the MMA kernels apply `1/scale`.
- Passing the docstring-style multipliers (`input_global_scale = a1_gscale`) returns all zeros.
- `check_small_m.py` → `results/small_m_L19.txt`: with the reciprocal-form scales, b12x is correct for M ≥ 4
  (rel-L2 vs CUTLASS 0.017-0.037), but returns **exactly zero output for M = 1, 2, 3**. Those sizes take the direct-micro
  kernel, whose docstring says "multiplier form". The result is the same with and without CUDA graphs.
- Unit scales (1.0) are the only setting that works for every M. This affects anyone who passes calibrated scales to
  `B12xMoEWrapper`/`b12x_fused_moe`.

## 5. b12x (fixed) vs CUTLASS speed

`bench_b12x_vs_cutlass.py` → `results/b12x_vs_cutlass_L19.txt`. Layer 19, real routing, CUDA graph, median of 30 replays.
Both sides include their bf16→FP4 input quantization. The router is excluded on both.

| M tokens | 3 | 12 | 48 | 192 | 1024 | 4096 | 8192 | 16384 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CUTLASS (us) | 32.5 | 49.5 | 179.5 | 256.5 | 321.7 | 550.8 | 870.6 | 1558.2 |
| b12x (us) | 23.5* | 57.0 | 157.7 | 293.1 | 309.6 | 435.4 | 631.1 | 1169.7 |
| ratio | 0.72* | 1.15 | 0.88 | 1.14 | 0.96 | 0.79 | **0.72** | 0.75 |

\* At M = 3 the output is wrong; see 4c.

The b12x results are mixed at decode sizes, and b12x is clearly faster from about 2k tokens upward. This is why the
patch is hybrid: b12x only for batches of 1024 tokens or more.

Kernel-level estimate for one 8k prefill pass (40 MoE layers at 8192 tokens): (870.6 − 631.1) us × 40 ≈ **9.6 ms**.

## 6. The patch: `03-moe-b12x-scale-fix-and-hybrid.patch`

The patch is a unified diff against pristine v0.32.0 files that are not in `build/`:
- `fused_moe/experts/flashinfer_b12x_moe.py`
- `fused_moe/oracle/unquantized.py`

Patched copies are in `patched/vllm/...` for bind-mounting. Everything is off unless `VLLM_B12X_FIX_SCALES=1`, and the
backend is only used with `--moe-backend flashinfer_b12x`.

- `VLLM_B12X_FIX_SCALES=1`:
  - no scale bake;
  - b12x gets `w1_alpha = w13_weight_scale_2`, `input_global_scale = a13` (= 1/a1_gscale), `fc2_input_scale = a2`,
    and `w2_alpha = w2_weight_scale_2 * a2`;
  - negative expert ids are masked (needed for b12x to run in vLLM at all);
  - the unquantized MTP MoE layer falls back to `auto` (FlashInfer CUTLASS bf16, as today).
- `VLLM_B12X_HYBRID_MIN_TOKENS` (default 1024): smaller batches run `flashinfer_cutlass_fused_moe` with exactly the
  FLASHINFER_CUTLASS backend's inputs: the same vLLM `scaled_fp4_quant`, the same swizzled scales, and the
  `g1`/`g2`/`a1_gscale`/`a2_gscale`. During vLLM's autotune pass the CUTLASS branch also runs, so its tactics are
  tuned for every bucket as today.
  - With the default threshold, b12x only ever runs its dynamic kernel (1024 tokens and up). The direct-micro scale
    mismatch from 4c cannot be reached.
  - Setting it to 0 (all b12x) requires `VLLM_B12X_ACT_SCALES=dynamic` (unit activation scales), because of 4c.
- Cost: b12x keeps a contiguous copy of the weight scale factors (about 50 MB per layer, 2 GB in total), which comes
  out of the KV-cache budget.

Run it with:

```
CONFIG=optimized FLAGS="-e VLLM_B12X_FIX_SCALES=1 \
  -v $PWD/kernels/moe-nvfp4/patched/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:ro \
  -v $PWD/kernels/moe-nvfp4/patched/vllm/model_executor/layers/fused_moe/oracle/unquantized.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/oracle/unquantized.py:ro" \
  scripts/serve.sh --moe-backend flashinfer_b12x
```

## 7. End-to-end (CONFIG=optimized server, `e2e.sh`)

Same harness, run back-to-back on the shared GPU. Columns:
- `cutlass` is the current optimized config.
- `hybrid` is the same config plus the patch, `--moe-backend flashinfer_b12x` and `VLLM_B12X_FIX_SCALES=1`.
- `hybrid2` is a rerun after moving the negative-id masking off the CUTLASS branch; its prefill numerics are identical to `hybrid`.
- `b12x as shipped` is the earlier rejected run (`results/quality/long_eval/b12x_fa8.json`).

The files are `results/e2e_<tag>.*`.

| metric | cutlass | hybrid | hybrid2 | b12x as shipped |
|---|---:|---:|---:|---:|
| NLL licenses 6.3k tok | 0.0909 | 0.1008 | 0.0989 | 0.1769 |
| NLL licenses 12.9k | 0.0802 | 0.0792 | 0.0818 | 0.1752 |
| NLL py_stdlib 6.9k | 0.5246 | 0.5305 | 0.5232 | 0.6224 |
| NLL py_stdlib 13.1k | 0.4774 | 0.4799 | 0.4767 | 0.5443 |
| NLL py_email 7.7k | 0.7935 | 0.7996 | 0.7950 | 0.8509 |
| NLL py_email 15.2k | 0.7823 | 0.7839 | 0.7864 | 0.8168 |
| needle (20) | 20/20 | 20/20 | 20/20 | 20/20 |
| MTP acceptance (decode bench) | 58.8 % | 60.0 % | 59.4 % | (lower; 658 vs 867 tok/s at c=4) |
| decode c=1 tok/s, median ITL | 323, 6.52 ms | 327, 6.71 ms | 350, 6.50 ms | |
| decode c=4 | 898, 8.16 ms | 751, 8.44 ms | 950, 8.19 ms | |
| decode c=16 | 2073, 14.14 ms | 1562, 14.51 ms | 1867, 13.89 ms | |
| decode c=64 | 3759, 28.76 ms | 3584, 29.28 ms | 3609, 28.83 ms | |
| **8k prefill TTFT mean / median** | **194.6 / 201.2 ms** | 188.2 / 192.2 | **182.8 / 187.0** | |

- **Quality:** the hybrid's mean NLL is 0.002-0.004 above `cutlass` over the 6 documents.
  - That is within run-to-run noise: two identical hybrid runs differ by up to 0.006, and this `cutlass` run differs
    from the earlier `v3c.json` run of the same config by up to 0.010.
  - The shipped b12x was 0.04-0.10 worse.
  - Needles and MTP acceptance are unchanged.
- **Decode:** decode batches are below 1024 tokens and run the same CUTLASS kernels as before, so median ITL is
  unchanged within ±0.3 ms.
  - The tok/s numbers at c=4 and c=16 swing ±15 % between runs of the same config; compare `cutlass` here with
    `results/decode/optimized.txt`: 871 / 1817.
  - The first hybrid run also launched 3 extra tiny kernels per layer for the id masking. hybrid2 removed them.
- **Prefill:** 8k TTFT is 6-12 ms lower (mean of the two hybrid runs is 185.5 vs 194.6 ms, -4.7 %). This matches the
  kernel-level estimate of 9.6 ms per 8k pass (40 layers × 240 us).
  - The TTFT includes scheduling. No nsys GPU-span run was done because the shared GPU queue held each run for 1-2 h.
- Not measured: 16k/32k TTFT and the throughput sweep.
- Memory: the b12x wrapper keeps about 2 GB of contiguous scale-factor copies plus its workspace, which reduces the
  KV cache.

## 8. Not done / ideas

- **CUTLASS glue (322 us of 895 us at 8k).**
  - `expandInputRows` runs at about 250 GB/s. Using 16 B vector loads and several rows per CTA should bring it near
    1.3 TB/s, saving about 100 us/layer, or 4 ms per 8k.
  - `doActivation` could be fused into GEMM1's epilogue (SwiGLU plus FP4 quant). That removes 67 MB of bf16 writes and
    reads per 4k tokens.
  - Both need a rebuild of FlashInfer's `fused_moe_120` JIT module from patched sources. With the hybrid b12x path,
    prefill no longer runs this code.
- **Hot experts.** One expert receives 73 % of the tokens in layer 19. Grouped-GEMM schedulers that split large groups
  across many CTAs matter more than the tile shape.
- **MTP layer.** Its unquantized bf16 MoE (`Fused_Moe_Kernel_sm80`, about 2.6 ms per 8k request in the old trace) is
  untouched. Quantizing it would change MTP acceptance.
- **Router.** Fusing the shared-expert gate into the router GEMM (N = 264) saves 9 us/layer at 8k. A tuned bf16 GEMM
  could approach 19 us from 33 us.
- **Upstream.**
  - vLLM: the b12x bake is the quality bug.
  - FlashInfer: the b12x direct-micro and MMA kernels disagree on the scale convention, and the documentation for
    `input_global_scale`/`fc2_input_scale` does not match what the MMA kernels do.

## Files and how to reproduce

All GPU scripts run in the vLLM image under the GPU lock via `run.sh`, with cwd `kernels/moe-nvfp4`.

| file | what |
|---|---|
| `capture_moe_inputs.py` | runs the full model once (eager) on an 8192-token real-text prompt (`data/prompt.txt`) and saves the MoE-block inputs of layers 0/9/19/29/39 to `data/moe_inputs.pt` (≈170 MB) |
| `moe_common.py` | loads a real layer, prepares it like vLLM's FLASHINFER_CUTLASS path, fp32 reference, routing |
| `check_b12x_scale_bake.py`, `check_b12x_bake_matmul_cpu.py` | CPU-only evidence for the bake bug |
| `numerics_b12x.py` | fused-MoE output errors (5 layers), b12x scale probes |
| `check_small_m.py` | M = 1 … 256 correctness, eager and CUDA graph |
| `bench_b12x_vs_cutlass.py` | speed per M |
| `bench_breakdown.py` | per-kernel breakdown and router GEMM |
| `bench_tactics.py` | tactic sweep vs autotuner |
| `e2e.sh <tag> <cutlass\|b12x_fixed>` | server: long_eval NLL plus needles, decode c=1/4/16/64, MTP acceptance, 8k TTFT |
| `patched/`, `orig/`, `03-moe-b12x-scale-fix-and-hybrid.patch` | the vLLM change |

```
# capture inputs (once, about 3 min of GPU)
flock $LOCK sg docker -c "docker run --rm --gpus device=0 --ipc=host --entrypoint python3 -e HF_HUB_OFFLINE=1 \
  -v /opt/intramind/hf-cache:/root/.cache/huggingface:ro -v $PWD:/work -w /work vllm/vllm-openai:v0.32.0 \
  kernels/moe-nvfp4/capture_moe_inputs.py 8192"
kernels/moe-nvfp4/run.sh batch_kernel_benches.sh      # numerics + speed + breakdown + tactics -> results/
kernels/moe-nvfp4/run.sh check_small_m.py 19
kernels/moe-nvfp4/e2e.sh hybrid b12x_fixed; kernels/moe-nvfp4/e2e.sh cutlass cutlass
```
