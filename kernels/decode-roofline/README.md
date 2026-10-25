# decode-roofline: how far MTP-2 decode is from the DRAM roofline, and one fix

Config measured: `CONFIG=optimized` (patches 01+02, MTP-2, async scheduling, CUDA graphs up to 16384),
random dataset 128 in / 256-512 out, on the RTX PRO 6000 (SM120).

**Short answer**
* **c=1: decode is not bandwidth bound.** One step must move about 5 GB, which takes 2.7-2.9 ms at
  1.8 TB/s. The GPU is busy about 6.5 ms per step, and host gaps are small (async scheduling hides the
  host). The excess comes from small-M kernel inefficiency and from launch count: about 1100 kernels per
  verify step, many of them 1-3 us glue kernels.
* **c=64: approximately bandwidth bound.** The byte floor is 22.7 ms at 1.8 TB/s (27 ms at an
  achievable 1.5 TB/s). The measured step is 28.7-29.2 ms with the GPU 100% busy. Two byte terms
  dominate: all 256 experts are read in every layer, and the fp32 GDN state is written three times per step.
* **Fix implemented (`VLLM_SKINNY_BF16_GEMM=1`, default off):** a Triton skinny BF16 GEMM for the router
  gate and GDN `in_proj_ba` at M<=16. On SM120, cuBLAS runs a 4-16-CTA wmma kernel for these
  (7.7 us in isolation, 11.5 us in-graph, 70 calls per verify step). Measured: the target verify graph drops
  from **4875 to 4292 us per step (-583 us, -12%)** at c=1. See section 4 for end-to-end numbers.

## 1. Bytes per decode step (from the real checkpoint tensor sizes)

`python3 -I bytes_model.py` (host only) reads the safetensors headers. It assumes ideal kernels: every
weight read once per forward, and only the distinct routed experts touched.

Facts that drive the model, checked in the checkpoint and the vLLM code:
* The **MTP layer is entirely BF16** (`ignore: re:^mtp.*`), including its 256 experts (6.29 MB per expert,
  against 1.77 MB for an NVFP4 expert). It **shares the FP8 lm_head** (`_maybe_share_lm_head`).
* MTP-2 runs **two drafter forwards and two full-vocab lm_head GEMMs per step**: pass 1 over the verified
  positions, then a 1-token pass. Both appear in the trace as CUDA graphs of 30-38 kernels, each containing a
  316 us lm_head GEMM.
* The GDN recurrent state is **fp32** (`mamba_ssm_dtype: float32`). Spec decode stores the state **after each
  of the 3 positions**: 1 read and 3 writes of 62.9 MB per sequence per step (FLA `fused_recurrent`, with
  the `INPLACE_FINAL_STATE` store inside the token loop; the CUDA `fused_gdn_decode_post_conv_mtp` has the same contract).

| c=1 (3 tokens verified) | MB | us at 1.8 TB/s |
|---|---:|---:|
| FP8 GDN projections (qkv, z, out) + bf16 a/b | 1015 | 564 |
| FP8 attention q/k/v/o (10 layers) | 273 | 152 |
| router + shared-expert gate (bf16) | 42 | 23 |
| NVFP4 shared experts | 71 | 39 |
| NVFP4 routed experts, ~20-23 distinct per layer | 1400-1646 | 777-915 |
| FP8 lm_head (verify) | 509 | 283 |
| GDN fp32 state: 1 read + 3 writes | 252 | 140 |
| KV cache (ctx 256), conv state | 9 | 5 |
| MTP bf16 dense, 2 passes | 157 | 87 |
| MTP bf16 routed experts | 167-197 | 93-109 |
| FP8 lm_head x2 (draft) | 1018 | 566 |
| **total** | **4.9-5.2 GB** | **2.7-2.9 ms** (3.3-3.5 ms at 1.5 TB/s) |

| c=64 (192 tokens verified) | MB | ms at 1.8 TB/s |
|---|---:|---:|
| NVFP4 routed experts (all 256 touched in every layer) | 18079 | 10.0 |
| GDN fp32 state: 1 read + 3 writes, 64 sequences | 16106 | 8.9 |
| MTP bf16 routed experts (nearly all touched) | 3007 | 1.7 |
| dense FP8 + 3 x lm_head + the rest | ~3630 | 2.0 |
| **total** | **~40.8 GB** | **22.7 ms** (27.2 ms at 1.5 TB/s) |

## 2. Measured step breakdown (nsys)

Traces: `traces/{base,skinny}_c{1,64}.sqlite` are graph-level and give accurate spans and host gaps.
`traces/node_c{1,64}.sqlite` use `--cuda-graph-trace=node` and give kernel names inside graphs; node tracing
inflates short kernels and host time. The analysis comes from `decode_analyze.py` and `step_categories.py`.
Medians are over 276-322 steps at c=1 and 87-93 full-batch steps at c=64.

| per decode step (graph-level trace) | base c=1 | skinny c=1 | base c=64 | skinny c=64 |
|---|---:|---:|---:|---:|
| step span (target-graph start to next) | 7159 us | 6029 us | 29190 us | 28692 us |
| GPU busy (union) | 6515 (91%) | 5884 (98%) | 29094 (100%) | 28600 (100%) |
| target verify graph | **4875** | **4292** | 24936 | 24483 |
| MTP drafter graphs (2 launches, include 2 x 316 us lm_head) | 1002 | 995 | 2487 | 2432 |
| eager kernels (verify lm_head 321-418 us, sampler) | 641 | 585 | 1676 | 1665 |
| idle (host/scheduler gaps) | 644 | 145 | 96 | 92 |

The c=64 target-graph difference (-453 us, 1.8%) is within run-to-run variation: the skinny path is not
taken at M=192. The c=1 idle difference mostly comes from nsys host overhead varying between runs. Host
gaps are small in every run, so **host/scheduler overhead is not the c=1 problem** (at most ~0.15-0.6 ms per step).

Target verify graph at c=1, by category (node trace, us per step, ~1030 kernels per step):

| category | us/step | kernels | byte floor | comment |
|---|---:|---:|---:|---|
| NVFP4 MoE grouped GEMM | 1227 | 80 | ~850 | ~1.2 TB/s effective |
| FP8 dense GEMM (GDN qkvz/out, attn qkv/o) | 1107 | 80 | 716 | 1.16 TB/s; out_proj/o_proj only launch 32 CTAs |
| **BF16 small GEMM (router 256x2048, in_proj_ba 64x2048)** | **808** | 70 | 25 | cuBLAS `cutlass_80_wmma 16x16`, grid 8x2 / 4x1 single-warp CTAs. **Fixed here** |
| MoE routing/permute glue (topkGating, BuildExpertMaps, expandInputRows, computeStrides, doActivation, cvt_fp4) | 760 | 320 | ~0 | 8 launch-bound kernels per layer |
| decode attention, FlashInfer XQA `kernel_mha` | 414 | 10 | ~5 | 41 us per layer at ctx ~300, grid 94x2 |
| norm/quant/act fused Triton | 308 | 191 | ~10 | launch bound |
| NVFP4 shared-expert GEMM | 306 | 80 | 39 | 0.23 TB/s |
| GDN recurrent decode (fp32 state) | 270 | 30 | 140 | 0.93 TB/s |
| gemv shared_expert_gate, conv1d, rope/kv-write, misc | 330 | 170 | ~5 | |

At c=64 the same graph is dominated by **GDN recurrent decode, 10.2 ms (353 us per layer, about 1.5 TB/s
on the 1R+3W state traffic)**, and the **NVFP4 MoE grouped GEMM, 10.1 ms**. Both are at or near
bandwidth. The rest is about 5 ms.

Eager work per step: verify lm_head 321 us at c=1 (1.59 TB/s, fine) and 418 us at c=64. The sampler
(`_topk_topp_kernel` + `_resample_kernel` + logits stats) costs 0.19-0.24 ms at c=1 and **0.95 ms at c=64**,
because the generation config sets top_k=20 and top_p=0.95. vLLM's Triton top-k/top-p runs one program per
row (3 CTAs at c=1), and with 248k-entry rows it reaches about 0.35 TB/s at c=64.

## 3. Gaps to the roofline, ranked (c=1, us per step, kernel time minus byte floor)

1. **MoE glue + grouped-GEMM inefficiency, ~1.1 ms.** 8 kernels per layer plus a grouped GEMM at 65-75% of bandwidth.
   An M<=16 "decode MoE" (top-k ids to NVFP4 gemv over the selected experts, fused activation and
   finalize, shared expert folded in as a 9th expert) would remove most of it, plus about 265 us of
   shared-expert inefficiency. This is the largest c=1 lever (owned by the MoE agent).
2. **BF16 small GEMMs, ~0.78 ms. Fixed by `VLLM_SKINNY_BF16_GEMM` (measured -583 us on the target graph).**
3. **FP8 dense GEMMs at M=3, ~0.39 ms** (dense-fp8-gemm agent): split-K for out_proj/o_proj, which launch only 32 CTAs.
4. **XQA decode attention, ~0.41 ms**: 41 us per layer for about 300 tokens of FP8 KV. A small-context
   path (no 94-way sequence split) or a FlashInfer decode kernel would make it ~5 us.
5. **Launch-bound Triton norm/quant and misc kernels, ~0.4 ms**: fusion (RMSNorm+quant into GEMM prologues).
6. **Sampler top-k/top-p, ~0.15 ms at c=1 and ~0.7 ms at c=64**: extend vLLM's split-row top-p
   pipeline to rows with an active top-k.
7. **GDN state traffic, 0.13 ms at c=1 and ~5 ms at c=64**: write 1 state instead of 3 (see section 5).
8. **Structural (not inefficiency): drafter lm_head 2 x 316 us and BF16 MTP experts.** Only removable by
   changing the draft model: a quantized MTP, or draft-vocab pruning.

## 4. The fix: skinny BF16 GEMM (patch `patch/03-skinny-bf16-gemm.patch`)

* `vllm/model_executor/layers/skinny_bf16_gemm.py` (new): a Triton kernel. Each program owns BN weight rows
  (BN chosen for >=128 programs) and streams them once over K, with up to 4 rows of x per grid row
  (M>4 is split over grid rows, so weight re-reads hit L2). fp32 accumulation and one bf16 rounding,
  the same contract as cuBLAS.
* `vllm/model_executor/layers/utils.py`: when `VLLM_SKINNY_BF16_GEMM=1`, `dispatch_unquantized_gemm`
  returns a wrapper that sends BF16 linears with 16<=N<=512 to a registered custom op
  (`torch.ops.vllm.skinny_bf16_linear`, opaque to torch.compile and safe under CUDA graphs). The op uses
  the skinny kernel only where it measured faster: M<=3, or M<=16 with N<=128. Otherwise it calls
  `F.linear`. Large linears never enter the op.
* Covered layers: the MoE router `GateLinear` (tier-6 fallback on SM120, since the specialized tiers are
  gated to SM90/SM100), GDN `in_proj_ba`, and the MTP router.

Microbench (`bench_skinny.py`, real weights, CUDA graph of 20 calls on distinct weight copies so reads
come from DRAM, median of 30):

| shape | M | cuBLAS us | skinny us |
|---|---:|---:|---:|
| router 256x2048 | 1 / 2 / 3 | 2.32 / 7.73 / 7.76 | 2.00 / 2.37 / 3.13 |
| router 256x2048 | 4 / 16 | 3.22 / 3.45 | 3.45 / 4.28 (not used) |
| in_proj_ba 64x2048 | 1 / 3 / 6 / 12 | 2.06 / 7.64 / 7.65 / 7.88 | 1.62 / 2.35 / 2.53 / 2.71 |
| in_proj_ba 64x2048 | 16 | 3.54 | 2.95 |

Correctness: the max abs error against an fp32 reference equals cuBLAS's error (for example 7.8e-3 on outputs
of magnitude 3.4, relative 2.3e-3, the bf16 rounding of the output). At M<=3, **99.9-100% of outputs are
bitwise identical to cuBLAS**. Strided input rows, bias and 3-D inputs are checked.

End to end (graph-level nsys, same session settings): **target verify graph 4875 to 4292 us per step at
c=1 (-583 us)**. GPU busy per step drops from 6515 to 5884 us. c=64 is unchanged.

Untraced A/B (`run_ab.sh`, a fresh server per side, CUDA graphs on as in production, `vllm bench serve` random
128 in / 512 out; `results/ab_{base,skinny}.log`):

| | base | skinny | change |
|---|---|---|---|
| greedy c=1 (16 prompts): tok/s, median ITL, acc. length | 401.0, 6.36 ms, 2.70 | 451.8, 5.77 ms, 2.78 | ITL **-0.59 ms (-9.3%)**; tokens/s per step-rate (tok/s / acc) +9.4% |
| default sampling c=1 (top_k 20, top_p 0.95, T=1) | 349.5, 6.57 ms, 2.42 | 376.1, 5.97 ms, 2.37 | ITL **-0.60 ms (-9.1%)**, tok/s +7.6% |
| greedy c=4 | 1117, 7.96 ms | 1115, 8.00 ms | none (noise) |
| greedy c=64 | 4665, 27.42 ms | 4637, 27.62 ms | none (noise; path not taken at M=192) |

**Estimated saving: ~0.6 ms per decode step at c=1 (about 9% of ITL), about 0 at c>=4.** Prefill is unaffected:
the op only takes the skinny path at M<=16, and prefill M is in the thousands.

Greedy probe (`eval/quality_probe.py`, 12 prompts x 160 tokens at T=0): this stack is **not deterministic run to run**.
The same base server answering the probe twice shares only 425/1393 identical-prefix tokens (max |dlogprob|
0.54 on the shared prefix), and skinny vs skinny gives 405/1507. Base vs skinny gives 308/1507 and 472/1395, the same
range, so the probe cannot resolve any difference. The evidence that numerics are unchanged is the kernel-level
check (outputs 99.9-100% bitwise equal to cuBLAS at M<=3, errors otherwise equal to cuBLAS's own error).
MTP acceptance length is unchanged within noise (2.70 vs 2.78 greedy, 2.42 vs 2.37 sampled).

## 5. What it would take to fix the c=64 floor (not implemented)

GDN spec-decode state: today every step writes the fp32 state after each of the 3 speculative positions
(1R + 3W, 537 MB per layer at c=64, 353 us per layer, 10.6 ms per step). Lossless alternative: keep one
committed base state plus the previous step's per-token (k, v, g, beta), about 50 KB per sequence per layer.
At the next step the kernel would load the base, replay the `num_accepted` tokens (the same fp32 ops in the
same order, so the result is bit-identical), write the new base, then process the 3 new tokens without storing
their states. That is 1R + 1W: **about -5 ms per step at c=64 (~18%)** and -0.07 ms at c=1. It needs:
(a) a variant of `fused_gdn_decode_post_conv_mtp` with a replay prologue; (b) a per-request (k,v,g,beta)
ring buffer in the mamba cache spec; (c) changes to vLLM's spec-block bookkeeping (`preprocess_mamba`,
the align-mode pre/post-copy kernels and prefix-cache checkpoints), which assume the state slots hold
materialized states. `--mamba-ssm-cache-dtype bfloat16` halves the same traffic by config alone, but it
**changes numerics**: the model ships `mamba_ssm_dtype=float32`, so it would need a quality evaluation.

## 6. Notes

* Draft-vocab pruning (FR-Spec style, a draft lm_head over the top-N tokens) would remove up to ~0.5 ms per step
  from the 2 x 316 us drafter lm_head. Output stays lossless, but it is **not viable without a frequency list
  taken from real traffic**: with this tokenizer, Vietnamese text lands at ids ~170k and Chinese at ~95-139k,
  so an id-prefix subset would wreck acceptance for this deployment's languages.
* The node-level trace (`--cuda-graph-trace=node`) inflates the c=1 step from ~6.5 to ~9.9 ms, mostly as host gaps.
  Use it for kernel names and durations only. Spans and idle time come from the graph-level traces.

## 7. Files

* `bytes_model.py`: the byte and roofline model (host).
* `serve_nsys.sh`: copy of `scripts/serve.sh` that passes `NSYS_OPTS` to `nsys launch`, writes traces to
  `traces/`, and never runs `build_patched.sh`.
* `capture.sh`, `run_capture.sh <tag> node|graph [skinny]`: one lock hold that starts the server, runs an
  untraced decode bench, the greedy probe and nsys windows at c=1/c=64, then removes the container.
* `run_ab.sh <tag> base|skinny`: untraced A/B (greedy c=1/4/64, default sampling c=1, probe twice).
* `decode_analyze.py`: per-step span, busy, graph/eager/idle split, a median-step timeline, per-kernel table.
* `step_categories.py`: category table for a node-level trace.
* `skinny_bf16_gemm.py`, `bench_skinny.py`: the kernel and its correctness/timing bench.
* `bench_lmhead.py`: FP8 lm_head GEMM bandwidth at decode M, full vocab and vocab prefixes. **Not run**:
  the lock time went to the A/B. The in-graph trace already shows 316-321 us for the 509 MB lm_head, i.e. ~1.6 TB/s.
* `compare_probe.py`: greedy-probe comparison (first divergence and logprob deltas).
* `vllm/model_executor/layers/{utils.py,skinny_bf16_gemm.py}`: patched files to bind-mount.
  `orig/`: unpatched utils.py. `patch/03-skinny-bf16-gemm.patch`: unified diff (a/vllm/... b/vllm/...).
* `results/`: decode benches and probes. `traces/`: nsys reports and sqlite.

## 8. Reproduce

```
cd /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/decode-roofline
python3 -I bytes_model.py
# kernel correctness + timing (GPU lock)
flock <gpu.lock> sg docker -c "docker run --rm --gpus device=0 --ipc=host --entrypoint python3 -e HF_HUB_OFFLINE=1 \
  -v /opt/intramind/hf-cache:/root/.cache/huggingface:ro -v $PWD/../..:/work -w /work vllm/vllm-openai:v0.32.0 \
  kernels/decode-roofline/bench_skinny.py"
# traces (each takes the lock itself, ~7 min)
./run_capture.sh base graph base ; ./run_capture.sh skinny graph skinny ; ./run_capture.sh node node
python3 -I decode_analyze.py traces/base_c1.sqlite ; python3 -I step_categories.py traces/node_c1.sqlite
# untraced A/B
./run_ab.sh ab_base base ; ./run_ab.sh ab_skinny skinny
python3 -I compare_probe.py results/ab_base_probeA.json results/ab_base_probeB.json
python3 -I compare_probe.py results/ab_base_probeA.json results/ab_skinny_probeA.json
```
To integrate: apply `patch/03-skinny-bf16-gemm.patch` on top of the image's vllm (it touches only
`model_executor/layers/utils.py`, which patches 01 and 02 do not touch), mount both files the way
`run_capture.sh` does, and set `VLLM_SKINNY_BF16_GEMM=1`.
