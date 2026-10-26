# fused-router: one kernel for router GEMM + scoring + top-k + normalisation (decode, 1-24 tokens)

Model: unsloth/Qwen3.6-35B-A3B-NVFP4-Fast (40 MoE layers + 1 MTP MoE layer, 256 experts, top-8, hidden 2048),
vLLM v0.32.0 image, RTX PRO 6000 Blackwell (SM120, 188 SMs, 128 MB L2). Config under test: MTP-2, c=1 verify step = 3 tokens.

## Answer

* **Is it worth it?  Yes at c=1 and c=2, marginal at c=4-8, no above 16 tokens.**
  In a CUDA graph of 40 layers with distinct real weights and L2 flushed between replays (section 5), the fused kernel costs
  **5.7 us per layer at M=3**, against 14.95 us for vLLM's production path (cuBLAS wmma router GEMM + `topkGating`) and 8.32 us for the
  already-built skinny Triton GEMM + `topkGating`. With 42 router calls per decode step (40 target layers + 2 drafter passes, assumed):

  | tokens M (concurrency) | vs production path | vs skinny-GEMM + topk | share of a ~6.0 ms step |
  |---|---:|---:|---:|
  | 1 | -48 us | -25 us | <1 % |
  | 3 (c=1, MTP-2 verify) | **-387 us** | **-109 us** | 6.4 % (5.4 % of the 7.16 ms base step) / 1.8 % |
  | 6 (c=2) | -77 us | -77 us | 1.3 % |
  | 12 (c=4) | -60 us | -60 us | 1.0 % |
  | 16 | -39 us | -39 us | 0.6 % |
  | 24 (c=8) | **+34 us (slower)** | +34 us | - |

  Of the 387 us at M=3, 277 us is the GEMM fix that `kernels/decode-roofline` already has (skinny kernel). **The extra from fusing is 109 us per step at
  c=1 (just above the 0.1 ms bar), 60-77 us at c=2-4.** Use it for M <= 16 and fall back above.
* **Upper bound before building** (section 2): removing the `topkGating` kernel completely costs 3.5-3.8 us x 42 = 147-161 us per step. The shared-expert gate
  (`gemvx` 2.1 us + `sigmoid` 1.1 us) sits on the aux stream with 43 us of slack, so fusing it gains nothing in latency.
  The kernel realises 109 us of the 147-161 us bound at M=3. The remaining ~2.5 us per layer is the cross-CTA hand-off, which is latency bound.
* **Numerics**: top-8 ids are identical to vLLM's `topk_softmax` on all injected-logit tests (random and tie-heavy, 6 shapes) and weights agree to <= 3.3e-7 relative
  (about 3 fp32 ulp; vLLM's kernel exp/sum order is closed source, bitwise weights are not reproducible). On real weights the fused kernel's bf16 logits equal
  the exact (fp64) logits rounded to bf16 in 99.985 % of entries and the top-8 set equals the exact routing in 16400 of 16400 rows. **vLLM's own cuBLAS path is the less exact one at M >= 4**
  (58.7 % of logits equal exact, 2.7 % of rows get a different top-8 set), so at M >= 4 the fused kernel changes the routing of ~3 % of rows relative to today (towards exact).
  At M <= 3 its top-8 sets equal the cuBLAS path in all 984 test rows. No end-to-end quality metric was run (section 8).
* **Not measured**: any end-to-end vLLM run. Everything is kernel-level in CUDA graphs. A draft vLLM patch is in `integration/` (compiles, applies cleanly, never run).

## 1. What vLLM v0.32.0 does between the hidden state and the NVFP4 grouped GEMM (read from the image sources)

`Qwen3NextSparseMoeBlock.forward` (`model_executor/models/qwen3_next.py`, also used by Qwen3.5/3.6-MoE) calls `self.experts(hidden_states, router_logits=hidden_states)`; the
`MoERunner` (`fused_moe/runner/moe_runner.py`) runs everything below inside the opaque custom op `moe_forward_shared`:

1. `SharedExperts.maybe_forward_async`: for <= 256 tokens the shared expert MLP is forked onto an aux CUDA stream (NVFP4 quant, gate_up GEMM, SiluAndMul, quant, down GEMM,
   then `sigmoid(shared_expert_gate(x)) * out` = `gemvx` + `sigmoid` + `mul`). It joins after the routed GEMM2.
2. `self.gate(hidden_states)`: `GateLinear`, 256 x 2048 bf16. The specialised tiers (cuteDSL low-latency GEMM, fp32 kernels) are gated to SM90 / SM100 (`is_blackwell` = family 100), and
   nobody calls `set_out_dtype`, so on SM120 it is tier 6, `F.linear` -> cuBLAS, **bf16 logits** (rounded once). With `VLLM_SKINNY_BF16_GEMM=1` it is the Triton skinny GEMM (decode-roofline).
3. `FusedTopKRouter` (no grouped top-k, no bias, `scoring_func="softmax"`, `renormalize = getattr(config, "norm_topk_prob", True)` = **True**, the HF config class has no such attribute):
   `ops.topk_softmax` -> kernel `vllm::moe::topkGating<8,256,...>` (1 warp). Softmax over all 256 experts in fp32 from the bf16 logits, top-8 with **lower expert index winning ties**, output sorted
   by score, then divided by the sum of the 8 selected probabilities. Padded rows (`VLLM_MOE_SKIP_PADDING`, default on) get `topk_ids = -1` and normal weights. Outputs: `topk_weights` fp32, `topk_ids` int32
   (+ a `token_expert_indices` buffer nobody reads). No EPLB, no expert map in this deployment.
4. FlashInfer CUTLASS experts (`FlashInferExperts`, TRT-LLM runner): vLLM `cvt_fp16_to_fp4`, then `fusedBuildExpertMapsSortFirstTokenKernel`, `expandInputRowsKernel`,
   `computeStridesTmaWarpSpecializedKernel`, grouped GEMM1, `doActivationKernel` (SwiGLU + FP4 quant), grouped GEMM2 with fused finalize.
   (The `_fse_fuse_gate` path in the runner that concatenates gate and shared-gate weights is ROCm-AITER only; on CUDA the shared gate lives inside `Qwen2MoeMLP`.)

Kernels of one layer at M=3 (nsys node trace `decode-roofline/traces/node_c1.sqlite`, 323 steps, median; node tracing inflates short kernels by ~0.5 us;
raw listing in `results/one_moe_layer_node_c1.txt`, script `layer_stats.py`, output `results/layer_stats_node_c1.txt`):

| stream | kernel | us |
|---|---|---:|
| main | fused add + RMSNorm (Triton) | 1.7 |
| main | **router GEMM, cuBLAS `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2`, grid 8x2 of 32-thread CTAs** | **11.68** |
| main | **`topkGating<8,256,4,16,32>`** (gap after the GEMM 0.10 us) | **3.84** |
| main | `cvt_fp16_to_fp4` (activation quant for the routed experts) | 1.3 |
| main | `fusedBuildExpertMapsSortFirstTokenKernel` | 3.4 |
| main | `expandInputRowsKernel` | 3.2 |
| main | `computeStridesTmaWarpSpecializedKernel` | 2.05 |
| main | grouped GEMM1 / `doActivationKernel` / grouped GEMM2 | 20.1 / 3.0 / 13.6 |
| aux | quant, shared gate_up GEMM, silu*mul, quant, down GEMM, **`gemvx` (shared_expert_gate) 2.11**, **`sigmoid` 1.1**, mul 2.0 | ~17 total |

Main-stream chain from router start to grouped GEMM1 start: 15.7 us (router span) + 8.0 us. The aux branch ends **43 us (median) before** the routed GEMM2 ends (p10 38, p90 48), so it is never on the critical path.
Topk kernels per step: 40 in the target graph (3.86 us mean) + 1 per drafter graph (3.2-3.7 us) = ~161 us per step in the trace. At c=64 (M=192) the same kernel takes 3.6 us (so it is a latency, not a work, cost).

## 2. Upper bound of fusing (step 2)

Per layer, M=3, from the trace and the CUDA-graph chain measurements (section 5; the chain numbers include ~0.85 us launch gap per kernel):

| piece | trace (node) | graph chain | per step (42 calls) |
|---|---:|---:|---:|
| cuBLAS router GEMM (production) | 11.68 | ~11.4 | ~480 us (the skinny patch removes ~275 of it) |
| skinny Triton GEMM | - | 5.12 | 215 us |
| `topkGating` | 3.84 | 3.58 | **147-161 us = bound for removing the top-k kernel** |
| shared-expert gate gemv + sigmoid | 3.2 (aux stream) | - | **0 (43 us slack)** |
| empty kernel in a captured chain | - | 0.82-0.87 | - |

The bound for "router GEMM + softmax + top-8 in one kernel" relative to the skinny-GEMM state is therefore ~150 us per step at M=3 (zero-cost tail), above the 0.1 ms bar, so the kernel was built.
Folding the shared-expert gate into the same kernel is implemented (extra weight rows appended after the router rows, output `sigmoid(bf16 logit)` as bf16, bitwise equal to `F.sigmoid(F.linear(x, w))` on 40 layers)
but is **not worth wiring in** for latency; it only removes 2 aux-stream kernels.

## 3. The kernel (`fused_router.cu`, `fused_router.py`)

Design taken from DeepGEMM `bf16_mega_gate` (read-only reference: swap-AB GEMM with experts on the MMA M side, split-K over CTAs, a per-token-block score barrier, one warp per token for the top-k,
optional bias, shared expert as extra expert, scaling factor). The SM100 specifics (UMMA/TMEM/TMA, 2-CTA pairs, persistent worker groups that spin on a co-resident barrier) are not usable on SM120, and spinning
on co-residency would deadlock when the shared-expert aux stream occupies SMs. This implementation:

* **Split-K variant (default)**: grid = (row groups x SPLIT, token blocks). 16 weight rows per group are the MMA M dimension, tokens are N (`mma.sync.m16n8k16` bf16 -> fp32, 1-4 n-tiles = up to 32 tokens per block).
  Weights and activations are loaded with 16-byte `ld.global.nc`; the k order inside a 32-wide step is permuted identically for A and B (legal for a dot product), so no shuffles or smem staging.
  Each CTA writes its fp32 partial logits to a scratch slot (deterministic sum order); the **last CTA to arrive** (release/acquire ticket, self-resetting so CUDA graphs replay) sums the splits in order and routes every token with one warp.
  No co-residency assumption (tested while another stream saturates the GPU).
* **Routing in the last CTA**: lane owns 8 consecutive experts. Fast path (softmax, no bias, bf16-rounded logits): ranking key = 16-bit order of the bf16 logit | inverted expert index in one uint32, so each pick is one `redux.sync.max`
  and ties go to the lower index by construction; weights `exp(l - max) * (1/sum)`, renormalised by the sequential sum of the 8 picks. General path (sigmoid / sqrtsoftplus / none, selection bias, fp32 logits): ordered-float `redux.max` + ballot.
  Verified equal to the fast path on softmax (ids exact, weights identical).
* **Parameters** (all runtime, no hard-coded model): `num_experts` (<= 512 split-K), `topk` (<= 32), `scoring` in {softmax, sigmoid, sqrtsoftplus, none}, `renormalize`, selection `bias` (select on score + bias, weight = unbiased score,
  DeepSeek-style), `scale` (routed scaling factor, applied like vLLM's `topk_sigmoid`), `round_logits` (bf16 rounding of the logits, what vLLM's GateLinear does), shared-gate rows (`S = rows(w) - num_experts`),
  `pad` mask (padded rows -> id -1), strided `x`, any `K % 32 == 0`, any `M` (token blocks of 32 beyond that), optional logits output.
  **Not covered**: grouped top-k (n_group / topk_group), expert maps / EPLB / EP, fp32 router weights (DeepSeek-V3 style fp32 gate), hash routing.
* `fused_router.py`: JIT build via `torch.utils.cpp_extension.load` (needs nvcc, present in the image; ~2 min first time, cached), `Workspace` (scratch + counters; one per stream is enough), `default_config`.
* **Cluster variant** (`mode="cluster"`): one cluster of 8 or 16 CTAs per token block, rows split across the CTAs, cross-CTA hand-off by distributed shared memory (E <= 256). Thread block clusters and DSMEM **work on SM120**
  up to 16 CTAs (`results/cluster_probe.txt`; a 2-barrier exchange costs +0.45 us over an empty kernel). It is correct, but slower than split-K at M <= 16 (section 5) because 16 SMs pull the 1 MB of cold weights
  slower than 34-136 CTAs do (6.5 us vs 5.7 us; 8 CTAs with twice the bytes each take 8.2-12.3 us, which fits a per-SM memory-parallelism limit, not isolated). It wins only at M=24 and ties cuBLAS there. Kept for reference, not recommended.

## 4. Correctness (`test_correctness.py`, `test_cluster.py`, `diag_precision.py`; results in `results/`)

GPU, vLLM 0.32.0 image, reference = vLLM's own `ops.topk_softmax` / `ops.topk_sigmoid` (and F.linear = cuBLAS for the GEMM part).

* **Routing arithmetic isolated** (logits injected exactly with an identity weight block, so both sides see the same bf16 logits): softmax E in {64,128,256,512}, k in {1,4,6,8}, M in {1,3,8,24}, random and
  tie-heavy logits (values in a 6-element set), renormalize on/off, plus the bias variants: 108 cases, **ids identical in all**, weights max rel diff 2.7e-7. Sigmoid + bias + scale (E=256/384), softmax + bias, no-renorm + scale=2,
  M up to 70 (3 token blocks): ids identical, weights <= 3.3e-7. Padding rows -> -1 like vLLM. Negative, all-zero and mixed-sign logits pass. sqrtsoftplus vs a torch reference passes.
  Tie behaviour was first measured on vLLM (`probe_vllm_topk.py`): ids `[5, 10, 100, 200, 7, 250, 0, 1]` for logits with equal values at those experts, i.e. lower index first, value-sorted.
* **Real weights, all 40 routers + MTP, RMSNorm-style activations with outlier channels, M = 1..32 (16400 rows)**: bf16 logits equal exact-rounded fp64 in 99.983-99.988 % of entries (cuBLAS: 99.88-99.99 % at M<=3 but **58.5-58.8 % at M >= 4**);
  top-8 set equal to the exact routing in **100 %** of rows (cuBLAS + vLLM: 97.3 % overall, 96.9-97.8 % at M>=4); on the 15825 rows whose 256 logits are all bit-exact, ordered ids are identical and
  weights agree to 1.2e-7 rel. The shared-gate sigmoid matches `F.sigmoid(F.linear(x, w))` bitwise. (The first version of this test compared against cuBLAS logits and "failed" on exactly this cuBLAS inexactness; v1 files kept.)
  `diag_precision.py` (4 layers x 48 rows): fused logits differ from round(exact) in <= 0.03 % of entries for split = 1..16 (cuBLAS: 0.13-0.26 %), max error 0.50-0.58 bf16 ulp, and chaining the tensor-core accumulator over K gives the same numbers as fresh accumulators (`accmode`), so `mma.sync` costs no precision here.
* Determinism (bitwise repeatable, independent of the release/acquire variant), CUDA graph replay with new data == eager, ticket counters back to 0, all `(nt, nw, split)` configs agree on the top-8 sets, strided `x`, L2 prefetch of
  the next weight does not change results, correct while another stream saturates the SMs.
* Totals in `results/correctness.txt`: **173 PASS, 0 FAIL**. Cluster variant (`results/correctness_cluster.txt`): **37 PASS, 0 FAIL** (cs 8/16 x nw 8/16 x M 1..45 with shared gate, E=128/64 ties, sigmoid+bias+scale, padding, real weights: top-8 set equals the exact routing in 2132 of 2132 rows, graph replay, concurrent load). The first run (`correctness_cluster_first_run.txt`) flagged one check that was mis-specified (it mixed the set comparison with a ulp bound); the test was fixed, not the kernel.

## 5. Microbenchmarks (CUDA graph, 40 layers, distinct real weights, L2 flushed with a 512 MB memset between replays; median of 40 replays; `bench_graph.py`, `results/bench_graph.txt`)

us per layer (graph replay time / 40). `cuBLAS+topk` = production (`F.linear` + `ops.topk_softmax`); `skinny+topk` = decode-roofline Triton GEMM for M <= 3 (its dispatch rule; cuBLAS above) + `ops.topk_softmax`.
Run-to-run repeatability of the baselines across my 3 runs: <= 0.1 us.

| M | cuBLAS+topk | skinny+topk | topk kernel alone | **fused split-K** | fused cluster (best cs) | saved vs best existing, us/layer | us/step (x42) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.76 | 6.20 | 3.53 | **5.61** | 6.50 | 0.59 | 25 |
| 2 | 14.95 | 6.86 | 3.58 | **5.68** | 6.66 | 1.18 | 50 |
| 3 | 14.95 | 8.32 | 3.58 | **5.73** | 6.71 | 2.59 | 109 |
| 4 | 7.83 | 7.83 | 3.64 | **5.79** | 6.76 | 2.04 | 86 |
| 6 | 7.83 | 7.83 | 3.64 | **5.99** | 6.91 | 1.84 | 77 |
| 8 | 7.88 | 7.88 | 3.64 | **6.09** | 7.01 | 1.79 | 75 |
| 12 | 7.94 | 7.94 | 3.69 | **6.50** | 7.68 | 1.44 | 60 |
| 16 | 8.04 | 8.04 | 3.69 | **7.12** | 7.88 | 0.92 | 39 |
| 24 | 8.29 | 8.29 | 3.74 | 9.11 | 8.29 | -0.82 | -34 |

Default config (`default_config`): M <= 4: 1 n-tile, 4 warps, split 4 (34-68 CTAs); M <= 6: 8 warps; M <= 12: 12 warps, split 2; larger: 16 warps, split 2. Few CTAs and a small split win because the hand-off, not the weight streaming, dominates.
The M=2/3 cuBLAS number (14.95) is the 11.5 us wmma kernel; at M >= 4 cuBLAS switches to a faster (but less accurate) kernel. The skinny GEMM alone is 3.0 / 3.7 / 5.1 us at M=1/2/3 in this harness.

Where the 5.5 us go (`bench_ablate.py`, `results/bench_ablate_v2.txt`, M=1, split-K (1,4,4), release/acquire ticket): empty-kernel floor 0.82; kernel stopped after the partial-logit stores **3.23**; + ticket 3.64; + partial reads 4.71; + scoring/top-k 5.53.
So ~2.4 us is weight streaming from cold DRAM/TLB (one round trip), ~1.5 us is the global hand-off, ~0.8 us is routing. A release/acquire ticket by one thread (CUTLASS-semaphore style) is 0.2 us faster than per-thread `__threadfence` + atomic.
The packed-key `redux` selection cut the routing part from 1.6 us to 0.8 us (`results/bench_ablate_v1.txt` vs `v2`).

Prefetch and gaps (`bench_prefetch.py`, `results/bench_prefetch.txt`): with a 12.9 us filler between layers (a 48 MB L2-resident streaming read), router cost per layer is skinny+topk 5.18 / 8.20 / 7.02 / 7.02 and fused 4.36 / 4.97 / 4.97 / 5.48 at M = 1 / 3 / 4 / 8.
Having the fused kernel `prefetch.global.L2::evict_last` the NEXT layer's router weight changes that to 4.77 / 4.77 / 4.77 / 4.79: -0.2 us at M=3, -0.7 us at M=8, **+0.4 us at M=1**; inconsistent, so it is off by default (`prefetch_next=`).
The benefit would need an end-to-end check because the real gap between two routers is ~60 us and 30+ MB of expert weights stream through L2 in between.

## 6. What is not measured, and caveats

* **No end-to-end vLLM number.** The 109 us / 387 us per step are microbench deltas times 42 calls. The drafter calls are assumed to behave like target-layer calls (M=3 and M=1); the real saving is whatever survives
  in-situ interactions (PDL work in `kernels/pdl-swapab` shrinks every kernel boundary, which shrinks the gain from removing one; the aux-stream shared expert competes for SMs with the 34-68 CTAs of this kernel).
* Cold-weight behaviour is emulated by a 512 MB memset between replays (48-60 % of the 5.7 us is cold DRAM/TLB latency); the real state between two routers differs.
* The benchmark chain has 40 routers back to back; in the model there are ~25 other kernels between them. The prefetch table above is the only measurement with a gap, and it is a stand-in.
* M > 32 (several token blocks) is correct but not benchmarked beyond M=24; it loses to cuBLAS there.
* Cluster variant: E <= 256 only. Split-K variant: E <= 512. Not tested: E > 512, K not a multiple of 32, fp32 weights, fp16/other dtypes, GPUs other than the RTX PRO 6000.
* Sigmoid / bias / sqrtsoftplus / scale paths are validated against vLLM's ops on injected logits, not on real DeepSeek weights, and without grouped top-k they are not a full DeepSeek-V3 router.
* The weights of the softmax are not bitwise equal to vLLM's (<= 3.3e-7 relative, ids exact); downstream they are consumed as fp32 by the finalize epilogue, so this is far below bf16 noise, but it is not zero.
* At M >= 4 the fused kernel is more exact than the production cuBLAS path, which changes ~3 % of rows' top-8 sets. Probably harmless (cuBLAS's own choice flips with M), but the quality probe on this stack cannot resolve it
  (see `decode-roofline/VERIFY.md`); an NLL run on a fixed prompt set with the flag on/off is the right test.
* The ticket / scratch design assumes launches on one stream do not overlap; give each stream its own `Workspace`.

## 7. Integration (draft, not run end to end)

`integration/vllm-moe-runner-fused-router.patch` (unified diff against v0.32.0 `fused_moe/runner/moe_runner.py`; checked with `py_compile` and `patch --dry-run` only): with `VLLM_FUSED_ROUTER=1` and `FUSED_ROUTER_DIR=<this dir>`
the runner skips `self.gate(...)` and `router.select_experts(...)` for <= 16 tokens and takes `(topk_weights, topk_ids)` from `fused_router(...)` when everything matches the Qwen path (unquantised bf16 gate without bias, `FusedTopKRouter` softmax,
no EPLB / capture hook / EP / DP / TP / SP, modular (non-monolithic) experts with int32 ids), otherwise falls through unchanged. The code lives inside the opaque `moe_forward` custom op, so torch.compile never traces it and the flag does not
touch the compile-cache key (the problem VERIFY.md found for `VLLM_SKINNY_BF16_GEMM`, which still has to be fixed separately). To run it, mount this directory, make nvcc visible (it is in the image) or pre-build the extension
into the shared `TORCH_EXTENSIONS_DIR`, and A/B at c=1/2/4 with the same harness as `run_ab.sh`. Not done here: only 1 GPU lock slot at a time, no vLLM server was started.
In-model the GDN `in_proj_ba` (64 x 2048) keeps using the skinny kernel; the fused kernel does not replace it.

Further ideas that were not built: fuse the activation FP4 quant (`cvt_fp16_to_fp4`, 1.3 us + gap per layer, 40 layers) into the same kernel because it reads the same hidden vector (needs the swizzled scale-factor layout of
FlashInfer's quant); prefetch the next layer's router weight from the previous MoE kernels; grouped top-k for DeepSeek-V3.

## 8. Files

* `fused_router.cu`, `fused_router.py`: kernel (split-K and cluster variants, benchmark-only `ablate`) and Python API. Build: `./run_gpu.sh` builds on first import (`TORCH_EXTENSIONS_DIR`, ~2 min); `build_only.py` builds without a GPU.
* `test_correctness.py`, `test_cluster.py`, `diag_precision.py`, `probe_vllm_topk.py`: correctness and numerics; `common.py`: real-weight loading.
* `bench_graph.py`, `bench_ablate.py`, `bench_prefetch.py`, `cluster_probe.cu` (build: `nvcc -O2 -std=c++17 -gencode=arch=compute_120,code=sm_120 -o cluster_probe cluster_probe.cu`): benchmarks; `run_gpu.sh <script>` runs one script in the image under the shared GPU lock (own container `fused-router-agent`, no ports); `run_all*.py` chain scripts in one lock hold.
* `layer_trace.py`, `layer_stats.py`: CPU analysis of the existing nsys node traces.
* `results/`: all raw outputs (`*_v1`, `*_v2`, `*_v3` are earlier kernel versions kept for the history: v1 tensor-core chain + strided partial loads, v2 packed-key selection, v3 + new default configs).
* `integration/`: draft vLLM patch.
