# pdl-swapab: two DeepGEMM ideas (PDL, swap-AB) for MTP-2 decode on SM120

Config: `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast`, vLLM v0.32.0 image, RTX PRO 6000 Blackwell (SM120, 188 SMs),
MTP-2, CUDA graphs. DeepGEMM only supports SM90/SM100, so only the ideas were used, no code.
Everything here is measured on the GPU or read from existing nsys traces; nothing was shipped or patched.

## Summary

| idea | verdict | size of the prize (c=1 step, ~6.4 ms) | what it would take |
|---|---|---|---|
| **A. PDL** | The idle gap inside the decode graph is already tiny (95 ns per kernel pair, 42 us of 4884 us). PDL only pays by hiding the *ramp* of the next kernel, and only if **both** kernels of a pair are PDL-aware. Measured: 0.15-0.3 us per small kernel pair, about 0.8 us per skinny GEMM that prefetches its weights into L2 before `griddepcontrol.wait`. FlashInfer's MoE and XQA kernels already use PDL (37% of the boundaries in the graph overlap today). | realistic **50-150 us (about 1-2%)**, ceiling about 210 us (3%) if every kernel were PDL. About 0 at c>=16 (steps of 14-29 ms). | Our GEMMs alone give nothing (+0.3%). The gain needs the neighbours too: inductor kernels (there is a switch, but see below, untested e2e) and our own GEMMs. |
| **B. swap-AB / smaller token tile** | FlashInfer's CUTLASS MoE **already is swap-AB** (decode tactics 16/17 and 56/57 = 128x32 tile, weights on the MMA M side, tokens on N=32). Shrinking the token tile further would save at most **3.3 us/layer = 0.13 ms/step (2%)** (extrapolated from the measured tile-size slope). The MoE decode kernel is byte-bound, not padding-bound. | tile shrink: <= 130 us. **But** b12x's small-batch kernels (CUDA-core `direct_micro` for M<=3, MMA `micro` for M<=5, no padding, no glue launches) are faster than CUTLASS with cold weights: **-34% at M=3, -16% at the padded c=1 size M=4 (38.7 -> 32.4 us/layer = 0.25 ms per step), -13% at M=8 (c=2, 0.32 ms per step)**, equal from M=12. | Not a new tile: a fused small-M MoE path. It already exists in FlashInfer; blockers are the activation-scale convention (kernels/moe-nvfp4 4c: use unit scales) and an end-to-end quality check. |

Recommended next step: section "Next step" at the end.

---------------------------------------------------------------------------------------------------

## A. Programmatic dependent launch

### A.1 Step 1: how big is the gap inside the decode graph? (no GPU, existing traces)

Source: `kernels/decode-roofline/traces/node_c1.sqlite` and `node_c64.sqlite` (nsys `--cuda-graph-trace=node`,
`CONFIG=optimized`, no skinny patch). Scripts: `gap_analysis.py`, `dur_hist.py`, `overlap_pairs.py`,
`pdl_candidates.py`, `pdl_edges.py`; outputs in `results/`. The node-level span of the target graph (4884 us)
matches the graph-level trace (4875 us), so node tracing does not inflate the inside of the graph.
Graph memset nodes are counted as nodes (41 per step, they are the MoE output zeroing).

Target verify graph, c=1 (323 launches, 1072 nodes = 1031 kernels + 41 memset/memcpy):

| metric | value |
|---|---|
| graph span (median) | 4884 us |
| sum of node durations | 5548 us (more than the span: the side stream and PDL overlap) |
| GPU idle inside the graph (span minus union of nodes) | **42 us (0.9%)** |
| boundaries (same-stream consecutive nodes) | 1070 per step |
| boundaries with a gap below 2 us | **all of them** (p50 0 ns, p90 96 ns, p99 320 ns) |
| boundaries that already overlap (= PDL in effect) | **400 of 1070 (37%)** |
| typical gap on a non-overlapped boundary | **95-100 ns** (the CUDA-graph kernel-to-kernel floor) |
| kernels shorter than 1.5 / 2.5 / 4 us | 300 / 483 / 758 (324 / 678 / 1526 us in total) |
| MTP drafter graphs (38 and 30 nodes) | gaps 6.6 and 5.6 us of 545 and 470 us |
| step-level idle outside the graphs (host/scheduler, from the decode-roofline README) | 145 us (skinny) to 644 us (base) per step, not addressable by PDL |

c=64 (109 launches, 1302 nodes): gap total **74 us of 24749 us (0.3%)**. PDL is irrelevant there.

The three largest per-pair gaps (us per step = pairs x mean gap): inductor quant kernel -> FP8 GEMM 11.2 us
(40 x 280 ns), `kernel_mha` (XQA) -> inductor 4.4 us (10 x 444 ns), and memset node boundaries 3.8 + 3.8 us
(`doActivation` -> memset -> MoE GEMM2, 40 x 95 ns each; the memset itself is 0.77 us x 40 = 31 us per step
and cannot be PDL-overlapped; zeroing the output inside an earlier kernel would remove it, but that is a
FlashInfer change). Everything else is the ~95 ns floor:

```
 3.8 us x40 95 ns  fp8_gemm -> triton(add+rmsnorm)       3.0 us x30 98 ns  bf16 wmma gemm -> triton(zeros)
 3.8 us x39 97 ns  moe grouped gemm -> triton(quant)      2.9 us x30 98 ns  conv1d -> gdn_decode
 3.8 us x40 94 ns  triton(quant) -> fp8_gemm              2.8 us x30 95 ns  gdn_decode -> triton(quant)
```

Who is already PDL (consumer side, `results/pdl_candidates_node_c1.txt`): MoE routing/permute glue 80% of its
200 boundaries overlap (365 ns average overlap), CUTLASS grouped GEMM 75% (436 ns), XQA 100%, conv1d 97%
(vLLM Triton `launch_pdl`), `cvt_fp16_to_fp4` 67%. FlashInfer passes `enable_pdl` = `device_support_pdl()` =
`major >= 9`, which is true on SM120, and the SM120 grouped-GEMM kernel (`sm120_gemm_tma_warpspecialized_*`)
has `wait_on_dependent_grids` / `launch_dependent_grids`. **PDL is therefore already on for the MoE chain and
for XQA.** Not PDL today: the 190 inductor Triton kernels (norm/quant/act), the 80 dense FP8 GEMMs (vLLM
CUTLASS `_C`), the 69 BF16 router/in_proj_ba GEMMs (cuBLAS), GDN decode, rope/kv-write, ATen elementwise, gemv.

### A.2 Step 2: what PDL can recover

**Microbenchmark** (`pdl_bench/pdl_chain.cu`, built with `build.sh`, run by `run_pdl_gpu_v2.sh` under the lock,
host process, no container). A CUDA graph of 160 dependent kernels: 16 layers of
`glue(amax+fp8 quant) -> skinny FP8 GEMM 12288x2048 -> glue -> 6 tiny 1-2 us kernels -> skinny FP8 GEMM
2048x4096`, M=4, 538 MB of distinct weights (4x the L2, so cold DRAM reads like a real step). The GEMM is a copy of
`kernels/dense-fp8-gemm/skinny/skinny_fp8.cu` with the PDL hooks. Launch attribute:
`cudaLaunchAttributeProgrammaticStreamSerialization` through `cudaLaunchKernelEx`, captured into the graph.
Inside kernels `griddepcontrol.wait` / `launch_dependents` (PTX). All modes give a **bitwise identical output**
(checksum column in the log). Median of 36 samples x 100 graph replays, two full runs agree to 0.1%.

| config (graph of 160 kernels, M=4) | us / graph | vs plain |
|---|---:|---:|
| plain | 473.0 | |
| our GEMMs PDL (wait + trigger), neighbours plain | 474.6 | +0.3% |
| our GEMMs PDL + weights prefetched to L2 before the wait, neighbours plain | 497.5 | **+5.2%** (worse) |
| glue/tiny kernels PDL only, GEMMs plain | 453.4 | **-4.2%** (-19.7 us) |
| everything PDL (wait + trigger) | 458.2 | -3.1% (-14.9 us) |
| everything PDL + weights prefetched to L2 (`cp.async.bulk.prefetch.L2`) before the wait | **428.0** | **-9.5% (-45 us)** |
| (M=16, same chain) plain / everything PDL + L2 prefetch | 476.1 / 445.0 | -6.5% |

Floor (600 dependent 4-CTA kernels): 0.693 us/kernel plain, **0.512 us/kernel PDL (-0.18 us each)**.

nsys node trace of the microbenchmark (`results/v2_chain_gap_summary.txt`): plain has **94 ns of gap** per
kernel pair (the same ~95 ns as in the real decode graph, 3.0% of span); with PDL the gap drops to 11-12 ns and
58 of 59 pairs overlap, but the span falls from 182.3 to 163.7 us (-18.6 us, 0.31 us per kernel). Only about
a third of that is the visible gap. The rest is the ramp of the next kernel, which PDL hides.

Reading:

* A PDL edge needs **both ends**: the consumer launched with the attribute *and* the producer calling
  `launch_dependents` early. Our GEMMs with plain neighbours get nothing (+0.3%), and prefetching weights
  there only adds traffic (+5%). So "launch our skinny kernels with PDL" is, on its own, worth 0.
* Per boundary: 0.15-0.18 us for tiny kernels, about 0.8 us per GEMM when the GEMM prefetches its weights into
  L2 while the producer runs ((45 - 19.7) us / 32 GEMMs). Prefetching registers (`ldg` before the wait) was
  not faster than the plain wait, and it exposed a real hazard: with `ld.global.nc` for the activations, ptxas
  hoisted the loads above `griddepcontrol.wait` and the output was wrong in one config (v1 run, kept in
  `results/v1_hoisting_bug/`). Activation loads after a wait must be plain `ld.global` with a memory clobber.
* **Inductor switch.** torch 2.13 has `TORCHINDUCTOR_ENABLE_PDL=1`: it emits `gdc_wait()` /
  `gdc_launch_dependents()` and `launch_pdl=True` into the generated Triton kernels (checked in the cache,
  `results/inductor_pdl_codegen_check.txt`: 3 of 3 kernels have it with the flag, 0 without). A test with
  inductor kernels alternating with cuBLAS GEMMs (`pdl_inductor.py`, 12 layers, M=4) gave **125.0 us with and
  without the flag, 0 overlapping kernel pairs**: exactly the "one end only" case above. The flag will only
  help where two PDL kernels are adjacent or the neighbouring GEMM is PDL-aware too. It has not been tested
  in the server (vLLM's compile cache key must also change; use a fresh `VLLM_CACHE_ROOT`).

**Upper bound and realistic estimate for the c=1 step.** `pdl_edges.py` counts the boundaries where both
kernels would be PDL-aware and that do not overlap today (per step of 1070 boundaries, 670 non-overlapped):

| scenario | new PDL edges / step | estimate |
|---|---:|---:|
| inductor flag only | 90 | 90 x 0.15 = ~14 us |
| our skinny GEMMs PDL-aware only | 70 | ~0 (producer not aware) |
| inductor flag + our GEMMs (FP8 + BF16, ~150 GEMM consumers with L2 prefetch) | 349 | 150 x 0.8 + 200 x 0.15 = **~150 us** |
| everything PDL (also cuBLAS, GDN decode, rope, gemv, ATen) | 589 | ~200 us (ceiling: 670 x 0.31 = 208 us) |

That is **1-2% of the 6.4 ms c=1 step realistically (up to 3% as a ceiling), and about 0 at c>=16.**
The estimate uses microbenchmark costs per boundary, not a model run; it is a bound, not a measurement.
The kernels we do not own (cuBLAS wmma/gemv, FlashInfer, vLLM `_C` quant, Triton from vLLM's own ops) cannot be
switched; the realistic scope is: inductor flag + our two skinny GEMMs (the FP8 one is CUDA, the BF16 one is
Triton, which supports `launch_pdl` and `tl.extra.cuda.gdc_wait`).

---------------------------------------------------------------------------------------------------

## B. Swap-AB / token tile for the NVFP4 MoE at decode

### B.1 What the two backends do at 1-24 tokens (source read in the vllm/vllm-openai:v0.32.0 image)

Model: 256 experts, top-8, hidden 2048, moe_intermediate 512. c sequences x 3 verified tokens (MTP-2). vLLM
pads the decode batch to the next CUDA-graph size (1 2 4 8 16 24 32 ...), so the MoE layer sees M = 4 for c=1,
8 for c=2, 16 for c=4, 24 for c=8, 48 for c=16 (the padded rows are routed too).

* **FlashInfer CUTLASS (the default, `FLASHINFER_CUTLASS`)**: `cutlass_fused_moe` -> `trtllm::fused_moe::gemm1/gemm2`.
  The SM120 grouped GEMM is a persistent warp-specialized kernel (grid 1x188, 384 threads, 87-93 KB smem). Tile
  table (`get_candidate_configs_sm120` x swapAB x finalize): 10 CTA tiles from 128x32x64B to 256x128x128B. Autotuned decode
  picks (`results/tactics_sweep_L19.txt`): GEMM1 tactic 16/17 = **128x32 + swapAB**, GEMM2 56/57 = 128x32 +
  finalize + swapAB. With `SwapAB=true` the launcher makes the weights the MMA A operand
  (`MainloopElementA = Weight`), so the **128-wide tile M is the weight rows and the 32-wide N is the token
  slot of one expert**. That is DeepGEMM's sm100 small-M choice. Minimum token tile in the candidate set: 32.
* **FlashInfer b12x (`--moe-backend flashinfer_b12x`)** puts tokens on M (no swap). `select_sm120_moe_backend`
  / `launch_sm120_moe`: routed rows < 32 and M <= 8 (i.e. **M <= 3 at top-8**) -> `direct_micro`, a CUDA-core
  FP4 dot-product kernel (`fp4_dot8_sum_f32acc`), no MMA, no padding, routing and quant fused in the same
  kernel, only for intermediate size <= 512 (this model: 512); rows <= 40 (M <= 5) -> MMA `micro`; rows <= 1024
  -> `static` (M tile 64 with a 32-row virtual-expert split, so at most 32 valid rows per tile); above that
  `dynamic` with M tile 16/32/64/128 chosen from rows per expert (`_select_dynamic_tile_m`).

### B.2 How much of the MMA work is padding

Measured routing (real layer 19 and the captured text, `tactics_sweep_L19.txt`): tokens per **active** expert
M=3: 24 rows / 23 active = 1.04; M=12: 96 / 70 = 1.4; M=48: 384 / 131 = 2.9; M=192: 1536 / 201 = 7.6.
With a 32-wide token tile the useful part of the MMA N dimension is **3%, 4%, 9%, 24%**. The rest is padding.

Byte/FLOP argument per layer at M=3 (23 experts x 3 x 512 x 2048 FP4 params):

* weights 23 x 1.77 MB = 41 MB -> 25 us at 1.6 TB/s;
* padded MMA 23 x 201 MFLOP = 4.6 GFLOP; useful 0.15 GFLOP. At the measured 2.2 PFLOPS NVFP4 peak the padded work is
  2.1 us, the useful 0.07 us. Arithmetic intensity is 3.5 FLOP/B per token per expert against a ridge of
  1200 FLOP/B, so the kernel needs about 340 tokens per expert to become compute bound;
* the kernel does **not** wait for the MMAs: the tactic sweep says that making the token tile 4x wider
  (128x32 -> 128x128, 4x the MMA work) moves GEMM1 by +5.6 us at M=3 (43.2 -> 48.8 us, default GEMM2) but by only
  **about +1% at M=192** (282.6 -> 286.3 us) and +0.2% at M=48 (182.5 -> 182.8 us), where bytes dominate.
  These sweeps replay one layer, so the weights are warm in L2; with cold weights the MMA share can only be smaller.

Extrapolating the M=3 slope backwards (0.06 us per padded token column for GEMM1, 0.08 for GEMM2) to a token tile
of 8 (the m16n8k64 minimum) saves about 1.4 + 1.9 = **3.3 us per layer, 130 us per step (2.7% of the 4.9 ms verify
graph)**. A tile of 16 saves about half. It is an upper bound for "same kernel, narrower tile", and it needs a
rebuilt `fused_moe_120` module (CUTLASS template change), since 128x32 is the smallest instantiated tile.

### B.3 Fraction of the MoE decode time

In-graph (c=1, node trace, `decode-roofline` categories): grouped GEMM1/2 = 1227 us per step (18.7 + 11.9 us per
layer, 40 layers), routing/permute glue 760 us (8 launch-bound kernels per layer), shared expert 306 us.
Bytes for the 21-23 distinct experts of the bytes model (37-41 MB, 23-25 us at 1.6 TB/s) against 30.6 us
measured: the grouped GEMMs run at 75-83% of the byte floor, so the headroom of **any** better grouped GEMM is
5-8 us/layer = 0.2-0.3 ms/step. Padding is ~3.3 us of
that, the rest is pipeline fill (8 k-iterations per tile), tail and launch ramp.

### B.4 Cold-weight measurement: CUTLASS swap-AB vs b12x

The existing microbenchmarks (`kernels/moe-nvfp4`) replay one layer, so its 41 MB of active weights sit in the
128 MB L2. `bench_cold_moe.py` (`run_cold_moe_gpu.sh`) calls 5 real layers (0, 9, 19, 29, 39) in one CUDA graph
with a 300 MB memset between calls (flush-only graph subtracted), real captured hidden states, real routing,
consecutive tokens in groups of 3 (what a verify step feeds). Both include their input quantization, router
excluded. b12x runs with unit activation scales (the only setting that is right for every M, moe-nvfp4 4c).
us per layer, cold:

| M (decode batch) | distinct experts / layer | bytes MB | byte floor @1.6 TB/s | CUTLASS (swap-AB 128x32) | b12x | b12x / CUTLASS | b12x saves per step (x40 layers) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 (c=1, unpadded) | 18.6 | 33 | 21 | 33.5 | **22.0** | **0.66** | 456 us |
| **4 (c=1 as vLLM runs it)** | 22.4 | 40 | 25 | 38.7 | **32.4** | **0.84** | **252 us** |
| 6 | 31.0 | 55 | 34 | 49.5 | 43.3 | 0.87 | |
| **8 (c=2)** | 40.2 | 71 | 45 | 60.8 | **52.7** | **0.87** | **324 us** |
| 12 | 57.0 | 101 | 63 | 79.5 | 82.7 | 1.04 | |
| 16 (c=4) | 68.4 | 121 | 76 | 92.8 | 90.3 | 0.97 | 100 us |
| 24 (c=8) | 86.4 | 153 | 96 | 115.4 | 112.2 | 0.97 | 130 us |
| 48 (c=16) | 136.4 | 241 | 151 | 174.1 | 170.9 | 0.98 | 130 us |

(M=3, 4, 8, 16, 24, 48 from `cold_moe_v3_padded.txt`, M=6 and 12 from `cold_moe_v2.txt`; the two runs repeat M=3, 24, 48
within 3%. Differences of 3% (about 3 us per layer) at M >= 16 are at the noise level.) The 4th token at M=4 is a
real token here, not the zero-hidden padding row vLLM adds; the padding row is routed too, so the real expert count
at c=1 is likely a bit higher than 22.

* At M=3 b12x (direct_micro) reaches 1.5 TB/s end to end, i.e. **the byte floor**, while CUTLASS is at 1.0 TB/s.
  The gain does not come from the tile shape but from a different design: no padding, no permute / activation /
  strides launches, one fused kernel. But vLLM runs c=1 at M=4 (routed rows 32, not < 32), where b12x switches to
  its MMA `micro` kernel: still **16% faster (6.3 us/layer = 0.25 ms per c=1 step, 4% of ITL)**; at M=8 (c=2) 13%
  faster (**0.32 ms per step**). The 2.5-3 us advantage at M >= 16 is within noise.
* From M=12 up both are 20-30% above the byte floor and equal. No tile shape would change that; the gap to the
  floor is fill/drain and launch.
* Numerics: b12x with unit activation scales differs from CUTLASS by rel-L2 0.06-0.16 per layer. The earlier
  report (moe-nvfp4 4b) shows both at 0.136 against an fp32 reference on layer 19 at 4096 tokens, so the difference is
  FP4-grid noise and not an error, but it was **not** checked at M <= 4, and it replaces the checkpoint's calibrated
  static activation scales by dynamic ones. It needs the NLL / MTP-acceptance check before anything is shipped.

---------------------------------------------------------------------------------------------------

## Next step

1. **B first, and it is a dispatch change, not a kernel**: in the existing hybrid patch (`kernels/moe-nvfp4`,
   `VLLM_B12X_HYBRID_MIN_TOKENS`, which now sends every batch below 1024 tokens to CUTLASS) route the padded decode
   sizes M = 4 and 8 (c = 1, 2) to b12x with unit activation scales and keep CUTLASS for 9..1023. Then NLL, needles and MTP
   acceptance (dynamic scales replace the calibrated ones, rel-L2 vs CUTLASS 0.06-0.16, parity with CUTLASS against
   fp32 only shown at 4096 tokens), and a decode A/B at c=1 and 2 (`run_ab.sh` style, fresh server per side).
   Expected: 0.25-0.3 ms of a 6.4-7 ms step (4%). Cheap to try; the risk is numerics, not speed.
2. Do not write a swap-AB / narrower-tile kernel for the CUTLASS grouped GEMM: at most 0.13 ms/step, needs a
   FlashInfer rebuild, and the b12x path already removes the padding and the glue.
3. PDL: only worth doing as a side effect. If the skinny FP8/BF16 GEMMs get a second revision, add the
   `launch_pdl`/`griddepcontrol` hooks and the L2 weight prefetch, and test `TORCHINDUCTOR_ENABLE_PDL=1` in the server
   (fresh compile cache) as a one-line A/B. Expect <= 1-2% at c=1 and nothing at c>=4. Do not expect gains from adding
   the attribute to our GEMMs alone.
4. Cheap find, not pursued: the 41 memset nodes (0.77 us each + 2 gaps) around MoE GEMM2 cost about 35 us per step
   and are not PDL-able; zeroing the output in `doActivation` would remove them (FlashInfer change).

## Files

* `gap_analysis.py` (idle gap per kernel pair, top neighbour pairs), `dur_hist.py` (duration histogram, union idle),
  `overlap_pairs.py` (which pairs overlap today), `pdl_candidates.py` (who controls each consumer),
  `pdl_edges.py` (PDL-edge scenarios). CPU only. Usage: `python3 -I <script> <node_trace.sqlite>`.
* `pdl_bench/pdl_chain.cu` (v2), `pdl_chain_v1.cu` (has the `ld.global.nc` hoisting hazard), `build.sh`,
  `run_pdl_gpu_v2.sh` (host process under the lock, ~1 minute of GPU), `pdl_inductor.py`, `run_inductor_gpu_v2.sh` (the PDL=0 vs 1 inductor test; v1 of it is `results/pdl_inductor_v1.txt`).
* `bench_cold_moe.py`, `run_cold_moe_gpu.sh` (uses a private copy of the FlashInfer JIT cache).
* `results/`: raw logs. `v2_*` is the valid PDL run, `v1_hoisting_bug/` the first run (all configs except
  "all PDL + weight regs" agree with v2), `cold_moe_v2.txt` (M=3,6,12,24,48) and `cold_moe_v3_padded.txt` (M=3,4,8,16,24,48) the MoE runs.

Reproduce: `./build.sh && ./run_pdl_gpu_v2.sh`; `./run_cold_moe_gpu.sh 3 4 8 16 24 48`; CPU analysis as above.
Limits: the PDL chain uses synthetic weights and an amax-quant stand-in for the Triton glue, one GPU, one clock
state (2850 MHz idle at start); the decode-graph gap numbers come from a node-level trace of the base config
(not the skinny patches); nothing was run end to end.
