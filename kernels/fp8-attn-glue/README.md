# fp8-attn-glue: remove the glue around the SM120 FP8 FMHA prefill (patch 01)

Area: `_sm120_fp8_paged_prefill` in `build/vllm/v1/attention/backends/flashinfer.py` and the
FlashInfer 0.7.0.post1 cute-DSL kernel `flashinfer/cute_dsl/attention/fmha/sm120/fmha_prefill_fp8_tma.py`.

## What patch 01 does per full-attention layer (10 layers plus the MTP draft layer = 11 calls per forward)

1. `k.index_select(0, idx)`, `v.index_select(0, idx)`: vLLM's FP8 cache is
   `[blocks, Hkv, page, 2D]` (K|V packed per row), and the compiled FlashInfer kernel takes
   compact `[pages, Hkv, page, D]` pools. So every page of the batch is copied, for every layer.
2. `query / q_scale` (skipped when `q_scale == 1.0`), `.clamp(-448, 448)`, `.to(fp8)`:
   two or three PyTorch elementwise kernels over a 64 MB BF16 Q at 8k.
3. The FMHA itself, q_tile 128, kv_tile 128, balanced causal scheduler.

## What changed (`glue_fmha.py`, 1 kernel launch in place of 4 or 5)

* **Strided K/V, no gather.** The kernel only reads K/V through TMA descriptors, and those are
  built at launch from the runtime tensor view. FlashInfer compiles the fake K/V tensors with
  static inner strides `(sym, page*D, D, 1)`. I compile them with symbolic, 16-byte-divisible
  strides instead (`GlueFMHA` + `_compile`), so the kernel reads vLLM's K and V views
  in place: HND `(Hkv*page*2D, page*2D, 2D, 1)`, or NHD after the transpose. The block table
  then holds absolute vLLM block ids (`absolute_block_table`). It is built once per step and
  shared by all attention layers of the step, like patch 01's metadata.
* **BF16 Q quantised inside the kernel.** The compute warps already load Q straight from
  global memory into MMA A-fragment registers, 4 elements per lane per row-quad. The variant
  loads 4 BF16 values (one 8-byte load) and converts them with
  `cvt.rn.satfinite.e4m3x2.f32` (`bf16x4_to_e4m3x4`). This matches
  `(q/q_scale).clamp(-448,448).to(e4m3)` bit for bit: BF16->F32 is exact, satfinite equals
  clamping to +-448 (inf included), and both sides round to nearest even. For
  `q_scale != 1`, PyTorch computes `bf16(x * fp32(1/q_scale))` (it divides by a CPU
  scalar through the reciprocal), and the kernel emulates that rounding step too.
  `q_scale` is a compile-time constant. This checkpoint has no `q_scale` tensor, so vLLM uses
  1.0.
* The rest of the kernel is unchanged: tiles, pipeline, softmax, epilogue. **The output is
  bitwise identical** to patch 01's path.
* `q_quant.py`: a one-kernel Triton Q quantiser, also bitwise exact. It is a fallback only;
  it is slower than doing the conversion inside the kernel (see below).

## Correctness (`correctness.py`, results/correctness.txt): ALL OK

* New path vs patch-01 path, varlen batches with cached prefixes and random page placement,
  HND and NHD layouts: single 8k; 8k chunk on an 8k prefix; 5 sequences 64..4097 with
  prefixes; tiny/odd lengths (1, 129, 300/301, 777, 2048/2111). **max abs diff = 0
  (bitwise)** in every case.
* A Q that holds every finite BF16 bit pattern, with a small softmax scale so every Q
  element affects the output, for q_scale 1.0 and 0.7071: bitwise equal output.
* The torch-emulation formula over all 65536 BF16 values (q_scale 1, 0.7071, 3): 0
  mismatches. Triton `q_quant`: 0 mismatches. A first version that went through Triton's
  `.to(float32)` and min/max flushed the 254 BF16 subnormals and was fixed.
* Against an FP32 reference on the same FP8 operands: max abs 1.9e-3 to 2.3e-3 with
  |out| max of about 0.13 to 0.16, identical for both paths (this is the FP8 P*V error the
  kernel already has).
* End to end in the server (`e2e.sh`): see "End-to-end" below.

## Timing (`bench.py`, results/bench_interleaved.txt)

Method: CUDA events, eager launches (vLLM runs prefill attention outside CUDA graphs in
piecewise mode), 256 MB L2 flush before every rep so Q and the K/V pages start cold, as they
do in the model. 5 warmups, median of 50 reps. The FMHA-sized variants run **interleaved
round-robin** because sequential blocks drifted by 5-10% with clock and temperature. Data:
Hq=16, Hkv=2, D=256, page 64, FP8 pool of 4096 blocks with random placement, BF16 output.

Per layer, microseconds:

| case | K+V gather | Q clamp+cast | FMHA only | **A: patch 01 total** | B: strided KV + torch quant | C: strided KV + Triton quant | **D: new (1 kernel)** | A-D | per request (x11 calls) |
|---|---|---|---|---|---|---|---|---|---|
| 8k (q=kv=8192) | 18.4+18.4 | 114.7 | 862 / 858* | **958.5** | 948.2 | 902.1 | **868.4** | **90** | **0.99 ms** |
| 16k | 26.6+28.7 | 290.8 | 3458 / 3452* | 3738.6 | 3710.0 | 3563.5 | 3466.2 | 272 | 3.0 ms |
| 32k (single pass) | 47+47 | 632.8 | 13747 / 13723* | 14411 | 14288 | 13941 | 13732 | 679 | 7.5 ms |
| 32k 2nd chunk (q=16k, kv=32k) | 49+49 | 290.8 | 10336 / 10316* | 10649 | 10570 | 10422 | 10325 | 324 | 3.6 ms (per chunk) |
| varlen 4 seqs (2048, 1024/5120, 3000, 2120) | 25+25 | 114.7 | 395 / 393* | 497.7 | 489.5 | 446.5 | 405.5 | 92 | 1.0 ms |

\* FMHA with pre-gathered compact KV (FlashInfer build) / with strided KV (new build) and
pre-quantised FP8 Q: reading K/V in place costs nothing. Converting Q inside the kernel costs
about 10 us at 8k (D minus the FMHA with FP8 Q) and is lost in noise at 16k and above. The
Q-quant kernel it replaces cost 115 us (torch) or 72 us (Triton) at 8k.

Note: the torch clamp+cast scales super-linearly (115 us at 8k, 291 at 16k, 633 at 32k).
At 8k the 64 MB clamp output still sits in the 128 MB L2 when the cast reads it; beyond that
it does not.

**Estimated savings: about 1.0 ms per 8k-token prefill** (11 attention calls per forward,
10 layers plus the MTP draft layer, x 90 us; the "optimized" config's GPU span is 132.7 ms,
so about 0.75%), about 7.5 ms per 32k prompt, and nothing for decode.
Spec-decode verify steps (q_len 3) never take this path (`VLLM_SM120_FP8_PREFILL_MIN_Q`=256).
It also removes 3 to 4 launches and two transient allocations per layer (K/V copies of
2x4 MB at 8k, 2x16 MB at 32k, plus the 32 MB FP8 Q).

## FMHA efficiency vs the ~1100 TFLOPS FP8 peak (task 4)

Causal FLOPs = 4*16*256*(Lq*Lkv - Lq^2/2). The kernel reaches **about 640 TFLOPS (58% of
peak)** at 8k, 16k and 32k, and 550 TFLOPS on the 4-sequence varlen batch (short sequences
leave more partially masked diagonal tiles and a ragged last wave).

Tile and scheduler sweep (results/fmha_tile_sweep.txt), all compiled with the same glue-free
kernel:

| q_tile/kv_tile | balanced=1 (8k / 16k / 32k TFLOPS) | balanced=0 |
|---|---|---|
| **128/128 (current)** | **664 / 672 / 641** | 574 / 605 / 621 |
| 128/64 | 476 / 453 / 451 | 402 / 424 / 439 |
| 64/128 | 346 / 341 / 344 | 316 / 329 / 339 |
| 64/64 | 498 / 457 / 455 | 455 / 441 / 447 |

The shipped config (128/128, balanced scheduler) is the best everywhere, so no tile change
is warranted. The balanced (longest-tile-first) scheduler is worth +4% at 32k and +16% at 8k.
Why it stops near 60%: the kernel is `mma.sync.m16n8k32` with 8 compute warps of 16 Q rows
each, and every warp reads the full 128x256 K tile and V tile from shared memory for each KV
tile. That is about 512 KB of smem reads per KV tile per CTA, roughly 4k cycles at 128 B/clk
against about 7.4k cycles of MMA at peak. Softmax (FP32 max/scale/sum, exp2, FP8 packing)
and the causal diagonal tiles add to that. Getting much closer to peak would need a
different warp tiling, for example 32 Q rows per warp so each K/V fragment feeds two MMAs.
The D=256 accumulator already sits at the register cliff (the kernel distributes row state
across the quad for this config), so that is a kernel rewrite, not a knob. Not attempted.

## Integration: `03-sm120-fp8-prefill-noglue.patch` (env flag, default off)

* A unified diff against `build/vllm` (`a/vllm/...`, `b/vllm/...`). It modifies
  `vllm/v1/attention/backends/flashinfer.py` and adds a new file,
  `vllm/v1/attention/backends/sm120_fp8_fmha_glue.py` (a copy of `glue_fmha.py`).
* Gate: `VLLM_SM120_FP8_PREFILL_NOGLUE=1`, on top of `VLLM_SM120_FP8_PREFILL=1`. The debug flag `VLLM_SM120_FP8_PREFILL_NOGLUE_VERIFY=1` also runs the old path and logs differences (slow). The fast
  path is used only when Q is contiguous BF16 and the K/V strides are 16-byte aligned. In
  every other case it falls through to patch 01's code.
* The compiled kernel is stored in FlashInfer's CuTe-DSL disk cache (module
  `sm120_fmha_fp8_glue`), so restarts do not recompile it (about 8 s on a cold cache).
* Serving: scripts/serve.sh mounts only the files patch 01/02 touch. To use this, mount the
  two `patch/b/...` files (or the files you get by applying the patch to `build/`) and add
  `-e VLLM_SM120_FP8_PREFILL_NOGLUE=1`. `e2e.sh` shows the exact mounts.

## End-to-end (`e2e.sh`, "optimized" serve flags + MTP-2, NOGLUE=0 vs 1)

Three server launches with the real weights: NOGLUE=0, NOGLUE=1, and NOGLUE=1 with
`VLLM_SM120_FP8_PREFILL_NOGLUE_VERIFY=1`. That last mode is debug only and syncs every layer.
It runs patch 01's path on the same real Q and KV cache next to the new path and logs the
number of differing output elements. Raw logs are in results/.

* **In-server bitwise check: 132 of 132 attention calls had 0 mismatched elements** (max
  abs 0). The calls covered: a single 9335-token prefill; a 4906-token chunk on a 9258-token
  context (chunked prefill); a varlen batch of 6 concurrent requests (15309 tokens); 313- and
  2561-token prefills; and the 64-request x 256-token warmup batch. Each forward makes **11
  calls, not 10**, because the MTP draft layer is also full attention and takes the same
  path.
* Greedy output and prompt logprobs differ between launches, but **the server is not
  deterministic even with NOGLUE fixed**. The same 9.3k-token prompt sent twice to the same
  server gives prompt-logprob mean |diff| 0.358 (NOGLUE=0) and 0.372 (NOGLUE=1), with an
  exact-equal fraction of 0. NOGLUE=0 vs NOGLUE=1 across launches gives 0.368, the same
  size. So the logprob comparison cannot resolve the change, and the bitwise in-server check
  above is the evidence. The nondeterminism comes from outside attention (for example MoE,
  GDN or batch-composition effects). It is worth a look by whoever owns those paths.
* 8k-token TTFT (vllm bench serve, 16 prompts, c=1, MTP-2): median 201.5 ms (NOGLUE=0) vs
  202.4 ms (NOGLUE=1); mean 199.5 vs 201.1 ms. That is **unchanged within noise**. The
  expected gain of about 1 ms (11 calls x 90 us) is below the run-to-run noise of this
  measurement (P99 is about 218 ms). An nsys GPU-span A/B would be needed to see it end to
  end; I did not run one.

## Reproduce

```
cd /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/fp8-attn-glue
./run.sh correctness.py          # bitwise checks, about 1 min (takes the GPU lock)
./run.sh bench.py --reps 50      # per-piece + interleaved A/B/C/D table
./run.sh bench.py --sweep        # adds the tile/scheduler sweep
./e2e.sh && python3 e2e_compare.py results   # two server launches, about 10 min under the lock
```
`run.sh` wraps the standard `flock ... sg docker -c "docker run ... vllm/vllm-openai:v0.32.0"`
command. It mounts the repo at /work and `kernels/fp8-attn-glue/.cache/flashinfer` as the
FlashInfer JIT cache.

## Files

* `glue_fmha.py`: `GlueFMHA` (a subclass of FlashInfer's kernel class that overrides only
  `load_q_tile`), `_compile` with strided K/V fakes, `get_kernel` (disk-cached),
  `absolute_block_table`
* `q_quant.py`: bitwise-exact single-kernel Triton Q quantiser (alternative; not used by the patch)
* `common.py`: vLLM-like pool, metadata, a replica of patch 01's glue, and an FP32 reference
* `correctness.py`, `bench.py`, `e2e.sh`, `e2e_client.py`, `e2e_compare.py`, `run.sh`
* `03-sm120-fp8-prefill-noglue.patch`, `patch/{a,b}/...`: patch sources
* `results/`: raw outputs
