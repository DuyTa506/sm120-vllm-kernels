# Dense FP8 GEMMs (per-channel W scale, per-token dynamic A scale) on SM120

Area: attention q/k/v/o, GDN in_proj_qkv / in_proj_z / out_proj and lm_head of
unsloth/Qwen3.6-35B-A3B-NVFP4-Fast under vLLM v0.32.0 on one RTX PRO 6000 Blackwell (SM120, 188 SMs).

**Short version**

* **Prefill (M = 8192 / 16384): no change shipped.** vLLM's CUTLASS kernel runs at about 690 TFLOPS in
  short bursts (63% of the 1100 TFLOPS register-resident peak). Under sustained load the board hits its
  600 W cap and the SM clock drops to about 1.9 GHz, and the same kernel then reaches **84% of the
  clock-scaled tensor peak**. None of the 12 other CUTLASS tile/schedule/raster configs I tried,
  cuBLASLt (`torch._scaled_mm` rowwise) or a different `kind::` MMA does better than -3% / +3%. All of
  them are bitwise identical to vLLM. Details are below.
* **Decode (M ≤ 64): a new skinny-M tensor-core GEMV beats vLLM's CUTLASS kernels by 1-3 µs per call.**
  It is shipped as a patch gated by `VLLM_SM120_SKINNY_FP8=1`. Per MTP-2 decode step that saves about
  **0.19 ms at c=1 and c=4 (about 3% of the 6.5 ms step)** and about 0.13 ms at c=16 (kernel-level
  estimate; end-to-end in the e2e section). The kernel reaches the DRAM floor: 1.45-1.67 TB/s on
  cold weights, versus 1.0-1.3 TB/s for vLLM on the small-N shapes.
* lm_head (248320 x 2048 FP8, 508 MB) is already close to bandwidth speed in vLLM: 1.62 TB/s, 315 µs.
  The skinny kernel gets 1.65 TB/s (309.5 µs). lm_head runs **3 times per decode step**: one verify call
  (M = 3c) plus 2 MTP draft calls (M = c), because the MTP head shares the FP8 lm_head. That is about
  0.95 ms (15%) of every c=1 step, and it is bandwidth-bound.

## 1. Shapes, call counts, dispatched kernels

| GEMM | N x K | calls / forward | weight MB |
|---|---|---|---|
| GDN in_proj_qkvz (qkv 8192 + z 4096, merged by vLLM) | 12288 x 2048 | 30 | 25.2 |
| GDN out_proj | 2048 x 4096 | 30 | 8.4 |
| attn qkv_proj (q+gate 8192, k 512, v 512) | 9216 x 2048 | 10 | 18.9 |
| attn o_proj | 2048 x 4096 | 10 | 8.4 |
| lm_head | 248320 x 2048 | 1 per logits call (3 per MTP-2 decode step) | 508.6 |

Dense FP8 weights read per decode step: 1.28 GB for the layers + 3 x 0.51 GB for lm_head.

vLLM path: `CutlassFP8ScaledMMLinearKernel` → `torch.ops._C.cutlass_scaled_mm` →
`cutlass_gemm_sm120_fp8_dispatch`. FlashInfer FP8 is rejected because it needs per-tensor scales. The
configs compiled into `_C_stable_libtorch.abi3.so` are listed below. I read them from the symbol table
and matched them against `scaled_mm_sm120_fp8_dispatch.cuh`:

| M | tile (MxNxK) | schedule | stages |
|---|---|---|---|
| ≤ 16 | 16x64x128 | Pingpong | 9 |
| ≤ 32 | 32x64x128 | Pingpong | 7 |
| ≤ 256 | 64x64x128 | Pingpong | 5 |
| > 256 | 128x128x128 | Cooperative | 2 |

All four use cluster 1x1x1, `PersistentTileSchedulerSm100` (CLC, full tile grid launched),
`Sm90TmaWarpSpecialized` epilogue with `D = bf16(sa[m] * (sb[n] * acc))`, and the
`mma.sync.kind::f8f6f4.m16n8k32 e4m3` MMA. `microbench/mma_kind_rate.cu` shows this MMA has the same
issue rate as the legacy e4m3 one: 2045 FLOP/clk/SM, about 1100 TFLOPS at 2.87 GHz.

## 2. Prefill (compute-bound)

`bench_sweep.py --ms 8192,16384`: CUDA graph, 2 weight copies, median of 30 replays, µs per call.
Logs are in `results/sweep_prefill_persistent.txt` (my harness with a persistent grid) and
`results/sweep_prefill_fullgrid.txt` (full grid, same as vLLM).

| shape, M=8192 | vLLM µs (TFLOPS) | best alternative | µs (TFLOPS) |
|---|---|---|---|
| gdn_qkvz | 597 (691) | CUTLASS 256x128x64 coop, swizzle 2 | 584 (707) |
| gdn_out | 199 (692) | 256x128x64 coop | 182 (754)* |
| attn_qkv | 452 (684) | 256x128x64 coop | 447-457 (677-692) |
| attn_o | 196 (702) | 256x128x64 coop | 179-187 (712-766)* |

\*At M=16384 the same config loses to vLLM by 0-3%, and the gain at M=8192 does not repeat between
runs (sweep 1 vs sweep 2). My compiled copy of vLLM's own 128x128x128 config (same CollectiveBuilder,
CUTLASS 4.5 headers from FlashInfer) is 1-7% *slower* than the kernel in vLLM's binary. Run-to-run and
build-to-build variance is therefore the same size as the differences between configs. Configs that
lose clearly: 128x128x64 coop (-5%), pingpong 64x128 / 128x64 / 64x256 (-10 to -20%), every persistent
grid (`sm_count` set) compared with the full CLC grid (-3 to -7%), and stream-K. cuBLASLt rowwise
(`torch._scaled_mm`) ties at M=8192 and is 4% slower at M=16384.

Why there is no headroom: `microbench/clock_under_gemm.py` runs the vLLM kernel back to back for 4 s.
It measures **611 TFLOPS at a mean SM clock of 1895 MHz and 591 W**. That is 84% of the clock-scaled
peak (2045 x 188 x 1.895 GHz = 729 TFLOPS). The kernel is power-limited, not latency- or
pipeline-limited. In a real prefill the GEMMs sit between lower-power kernels, so the clock is
somewhere between the two cases. Total dense-GEMM time per 8k prefill is about 30 ms
(30 x (597 + 199) + 10 x (452 + 196) µs). A tile change that saves 1-3% (0.3-0.9 ms) would need
CUTLASS JIT compilation at server start, and the gain is within build-to-build noise. **Not shipped.**

The fused rms_norm + per-token quant Triton kernels from the old trace (about 41 µs for 4.6k x 2048)
already move about 1.5 TB/s (bf16 in, residual read+write, fp8 out). Folding the quant into the GEMM
prologue is the only remaining lever, and it is outside the scope of a dispatch change.

## 3. Decode (bandwidth-bound): skinny tensor-core GEMV

Kernel: `skinny/skinny_fp8.cu`.

* **Swap-AB layout.** The weight is the A operand of `mma.m16n8k32 e4m3` (16 output channels per
  warp). The tokens are the B operand (8 per n8 tile, up to 8 tiles, so M ≤ 64).
* **No shared-memory staging.** Each lane issues 16-byte `ld.global.nc.L1::no_allocate` loads straight
  into mma fragments. K is permuted inside each 128-wide k-block so that every load is a contiguous
  16 B. The same permutation is applied to the activations, and a dot product does not depend on K
  order, so the result is unchanged.
* **Split-K inside the block.** The block's warps (`kw`) split K. The partial sums are reduced in
  shared memory in a fixed order, so the result is deterministic and the same for every batch size.
* **Epilogue.** `bf16_rn(sa[m] * (sb[n] * acc))`, the same operation order and rounding as vLLM's
  CUTLASS epilogue.

Timing (`bench_sweep.py --ms 1,3,4,12,16,48,64,192`, log in `results/sweep_decode_cold.txt`): CUDA
graph of R distinct weight copies totalling 640 MB or more, so weights are read cold as in a real
forward (L2 is 128 MB). Median of 30 replays, µs per call.

| shape | M | vLLM µs (GB/s) | skinny µs (GB/s) | saved µs |
|---|---|---|---|---|
| gdn_qkvz | 4 | 20.2 (1252) | 17.2 (1469) | 3.0 |
| gdn_qkvz | 16 | 20.1 (1271) | 17.0 (1505) | 3.1 |
| gdn_qkvz | 48 | 21.2 | 19.4 | 1.8 |
| gdn_out | 4 | 8.4 (997) | 6.6 (1267) | 1.8 |
| gdn_out | 16 | 8.4 | 6.7 | 1.7 |
| gdn_out | 48 / 64 | 9.8 / 9.9 | 8.7 / 8.3 | 1.1 / 1.6 |
| attn_qkv | 4 | 14.3 (1325) | 13.2 (1434) | 1.1 |
| attn_qkv | 16 | 14.3 | 13.2 | 1.1 |
| attn_qkv | 48 / 64 | 14.6 | 15.4 / 17.0 | slower → vLLM kept |
| attn_o | 4 | 8.4 | 6.6 | 1.8 |
| attn_o | 48 / 64 | 9.8 | 8.6 / 8.3 | 1.2 / 1.5 |
| lm_head | 1 | 314.2 (1620) | 310.2 (1641) | 4.0 |
| lm_head | 3 | 315.3 (1617) | 309.5 (1648) | 5.8 |
| lm_head | 12 / 16 | 318.4 / 319.6 | 309.7 / 309.8 | 8.7 / 9.8 |
| lm_head | 48 | 325.0 | 318.3 | 6.7 |
| lm_head | 64 / 192 | 324.6 / 456.5 | 324.4 / n.a. | 0 → vLLM kept |

The (kw, kb) choice per shape is in `pick()` in `skinny/dispatch.py`, mirrored in the patch. Shapes
where it returns `None` keep vLLM's kernel.

Estimated saving per decode step (MTP-2: the layers see the CUDA-graph-padded M = 4 / 16 / 48 / 192,
lm_head sees M = 3c once and M = c twice):

| concurrency | layer GEMMs | lm_head | total / step |
|---|---|---|---|
| c=1 (step ≈ 6.5 ms) | 30·3.0 + 30·1.8 + 10·1.1 + 10·1.8 = 173 µs | 5.8 + 2·4.0 = 14 µs | **≈ 0.19 ms (2.9%)** |
| c=4 | 30·3.1 + 30·1.7 + 10·1.1 + 10·1.8 = 173 µs | 8.7 + 2·5.8 = 20 µs | **≈ 0.19 ms** |
| c=16 | 30·1.8 + 30·1.1 + 0 + 10·1.2 = 99 µs | 6.7 + 2·9.8 = 26 µs | **≈ 0.13 ms** |
| c=64 | 0 (M=192 stays on vLLM) | 0 | 0 |

The kernel is close to the floor. For example, gdn_out moves 8.4 MB in 6.6 µs, of which about 5.1 µs
is the transfer at 1.65 TB/s and the rest is launch plus first-byte latency. Going further needs
programmatic dependent launch or fusion with the neighbouring kernels, not a better GEMM.

Not shipped, but found: for M = 192 (c=64), CUTLASS 64x128x128 pingpong (bitwise identical) beats
vLLM's 64x64x128 on lm_head (431.6 vs 456.5 µs) and gdn_qkvz (23.0 vs 24.2 µs). That would be a
one-line change in vLLM's C++ `sm120_fp8_config_M64` for N ≥ 12288. It is not reachable from Python
without JIT-compiling CUTLASS at server start.

## 4. Correctness

`correctness.py --impl skinny` uses real weights of layers 0, 1, 2, 3, 7, 38, 39 and lm_head, with LLM-like
activations (lognormal channel scales and 20x outlier channels, per-token FP8 quant). It covers M in
{1, 2, 3, 5, 7, 8, 9, 12, 15, 16, 17, 24, 31, 33, 48, 63, 64}, odd sizes included. Results:
255 (shape, layer, M) cases (`results/correctness_and_patch_test.txt`):

* Bitwise equal to vLLM's output for **99.95% of elements at worst and 99.99% on average**.
* The elements that differ are one-ulp bf16 rounding flips caused by the different fp32 summation
  order. The largest difference is 0.031 on values of magnitude 8-16. Bit-pattern distances of up to
  203 ulp occur only on near-zero outputs.
* Max abs error vs the fp32 reference is **identical to vLLM's in every case** (worst relative
  3.69e-3 for both, which is the bf16 output rounding). The skinny kernel is never less accurate.
* For a fixed token row, the result does not depend on the batch size.

`test_patch_op.py` runs with the patch files bind-mounted and `VLLM_SM120_SKINNY_FP8=1`. It checks the
JIT build, the op against `cutlass_scaled_mm` on all shapes for M in {1 ... 1000} (falls back bitwise
above the table), the op under `torch.compile(dynamic=True)`, and CUDA graph capture/replay:
**PASS**. The op matches eager mode exactly under torch.compile for M = 4, 16, 48 and 300, and CUDA
graph replay matches exactly. M = 65, 192 and 1000, and every shape the table excludes, return vLLM's
bits exactly.

## 5. End-to-end

RESULTS_E2E

## 6. Patch

`sm120-skinny-fp8-gemm.patch` uses a/vllm, b/vllm paths against the stock v0.32.0 image. None of the
touched files are in `build/vllm` yet.

* `vllm/model_executor/kernels/linear/scaled_mm/cutlass.py`: `CutlassFP8ScaledMMLinearKernel` decides
  once, at weight-processing time, whether a layer qualifies. It qualifies when the flag is on, the
  device is SM12x, the activations are dynamic per-token, the weight scales are per-channel fp32, no
  padding is needed and the output is bf16. Qualifying layers call `torch.ops.vllm.sm120_fp8_scaled_mm`.
* `.../sm120_skinny_fp8.py` (new): the custom op (with a fake impl for torch.compile) and the tuned
  `_pick` table. The M decision is made inside the op at run time, so there is no symbolic-shape guard,
  and it falls back to `torch.ops._C.cutlass_scaled_mm` (bitwise vLLM) for everything else. It
  JIT-compiles the kernel with nvcc into `$VLLM_CACHE_ROOT/sm120_skinny_fp8`: about 1 minute the first
  time, then cached.
* `.../sm120_skinny_fp8.cu` (new): the kernel, with only the 3 (kw, kb) variants the table uses.

To use it with `scripts/serve.sh`, add `-e VLLM_SM120_SKINNY_FP8=1` and bind-mount the three patched
files over `$PKG/model_executor/kernels/linear/scaled_mm/` (see `e2e_decode.sh`). Optional:
`VLLM_SM120_SKINNY_FP8_MAX_M` (default 64).

## 7. Reproduce

All GPU commands take the shared lock (`run_gpu.sh` wraps flock + docker + an in-container timeout).

```
cd /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels
# build the extensions (CPU only, no lock): CUTLASS config sweep + skinny kernel
sg docker -c "docker run --rm -u 1000:1000 -e HOME=/tmp --entrypoint python3 -v $PWD:/work -w /work/kernels/dense-fp8-gemm vllm/vllm-openai:v0.32.0 -c 'import sys; sys.path.insert(0,\".\"); import cutlass_ext.build as c, skinny.build as s; c.load(); s.load()'"
kernels/dense-fp8-gemm/run_gpu.sh kernels/dense-fp8-gemm/bench_sweep.py --ms 1,3,4,12,16,48,64,192      # decode sweep
kernels/dense-fp8-gemm/run_gpu.sh kernels/dense-fp8-gemm/bench_sweep.py --ms 8192,16384 --shapes gdn_qkvz,gdn_out,attn_qkv,attn_o
python3 kernels/dense-fp8-gemm/summarize.py <log> 4
kernels/dense-fp8-gemm/run_gpu.sh kernels/dense-fp8-gemm/correctness.py --impl skinny
kernels/dense-fp8-gemm/run_gpu.sh kernels/dense-fp8-gemm/microbench/clock_under_gemm.py
kernels/dense-fp8-gemm/e2e_decode.sh off; kernels/dense-fp8-gemm/e2e_decode.sh on
```

Files: `common.py` (weights, activations, timing), `bench_sweep.py`, `bench_baseline.py`,
`summarize.py`, `correctness.py`, `test_patch_op.py`, `cutlass_ext/` (templated SM120 FP8 CUTLASS GEMM
with vLLM's epilogue, used for the config sweep), `skinny/` (kernel, build, dispatch), `microbench/`
(MMA rate and clock/power probes), `patch/` (a/ b/ trees), `results/` (raw logs).
