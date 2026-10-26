# Independent verification: SM120 skinny FP8 GEMM (VLLM_SM120_SKINNY_FP8)

Verifier's notes. I did not modify the author's code. My files are `verify_kernel.py` and `verify_insitu.py`
in this directory. Logs, nsys reports and runner scripts are in the verifier scratchpad
(`/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/verify-fp8/`).

**Verdict: PARTIALLY confirmed.**

* The kernel is correct, and it is faster at small M. My numbers match or beat the author's microbenchmarks.
* The real per-step gain is smaller than claimed. At c=1 it is about 0.15 ms (claimed 0.19 ms). At c=4 it is
  about 0.06 ms (claimed 0.19 ms), because c=4 decode really runs at M=18, not M=16.
* There is a compile-cache integration hazard. If the flag is toggled with the patched files in place, the
  change is silently disabled for all 80 layer GEMMs.
* The end-to-end ITL gain claimed by the author (-0.11 to -0.17 ms at c=1) is the same size as the wall-clock
  noise I measured on this shared host. It is consistent with the kernel savings, but it does not prove them
  on its own.

## 1. What I checked in the code

* The patch file, `patch/b/...` and `skinny/skinny_fp8.cu` contain the same kernel. The vLLM copy only
  instantiates the 3 (kw, kb) variants the table uses. The `_pick` table is identical in `skinny/dispatch.py`
  and in the patch.
* The epilogue is `bf16_rn(sa[m] * (sb[n] * acc))`, the same order as CUTLASS ScaledEpilogue. Only the fp32
  K-summation order differs.
* The gate is checked once per layer (per-token dynamic activations, fp32 per-channel `[N,1]` scales,
  `[K,N]` col-major e4m3 weight, `K % 1024 == 0`, `N % 16 == 0`, bf16 out, no bias). The checkpoint's bf16
  `weight_scale` is loaded into fp32 parameters by compressed-tensors, so the gate does pass for this model.
  I confirmed this in-model with nsys.
* At run time the op checks `M <= MAX_M`, `_pick != None`, `sa.numel() == M`, a contiguous A and a 16 B
  aligned A. In every other case it falls back to `torch.ops._C.cutlass_scaled_mm`.
* No vLLM compile pass pattern-matches `cutlass_scaled_mm`. I grepped `vllm/compilation`. Hiding it behind a
  new custom op therefore loses no fusion. Inductor still fuses the per-token quant into the preceding Triton
  kernels in both cases.
* The MTP layer's own projections are BF16 in the checkpoint and are untouched. Only the shared FP8 lm_head
  is affected on the drafter side.
* Workload equality in the author's bench is fine. Both sides run inside one CUDA graph, use the same
  activations and the same rotated weight copies, and are timed as the median of 30 replays.

## 2. Correctness (verify_kernel.py, Part A): CONFIRMED, with one claim corrected

I used real weights of layers the author did not test (GDN 4, 5, 22, 33; attention 11, 35; lm_head), fresh
activation seeds, every M from 1 to 64, plus M = 65, 96 and 192. The shipped op was compared against the
stock `ops.cutlass_scaled_mm` (what stock `apply_scaled_mm` calls).

| metric | result |
|---|---|
| bitwise-equal elements vs stock, worst case | 99.935 % |
| max abs diff vs stock | 0.0625 (bf16 ulp flips on large values; the author reported 0.031 on their set) |
| max rel err vs fp32 reference, worst case | 3.80e-3 patched, 3.80e-3 stock (identical) |
| mean abs err vs fp32 ref, patched worse than stock by >1 % | 0 of 469 cases (identical to 3 digits everywhere) |
| fallback cases (M > 64, or `_pick` = None, e.g. attn_qkv M 17-64, lm_head M 49-64) | `torch.equal` to stock in every case |
| run-to-run determinism (5 reruns, M=48) | yes, patched and stock |
| non-contiguous A | falls back, identical to stock |
| M = 0 | raises (stock raises too, `cutlass_gemm_caller ... Error Internal`), so no regression |
| A with an 8 B misaligned start | the op falls back to stock, and stock itself fails the TMA descriptor and then hits an illegal instruction. This is pre-existing, not a regression. |
| saturated A (all ±448, scale 3) | max diff vs stock 0.25 on large outputs (ulp level; I did not quantify it relative to output magnitude) |

**Correction: "batch-invariant per row" is only true inside one config.** For gdn_qkvz, attn_qkv and
lm_head, row r computed at M=48 differs from the same row computed at M=1. The table switches kw 1 → 2
at M=17 for gdn_qkvz, and skinny → stock for attn_qkv and lm_head. Stock CUTLASS is batch-invariant in the
same test (M=48 vs M=1 equal for all shapes). gdn_out and attn_o (kw=8 throughout) stay invariant. This is
minor (vLLM is not batch-invariant elsewhere either), but the claim as written is wrong, and it matters for
anyone relying on stock-GEMM batch invariance.

Token level: greedy outputs on random-token prompts diverge between off and on within the first few tokens.
They diverge just as early between two **stock** runs (off1 vs off2: first divergence at token 0-4), because
random-token prompts have near-tied logits. So the patch's drift cannot be told apart from stock run-to-run
nondeterminism. Nobody (author or verifier) ran an accuracy eval. That is acceptable for a change that is
bf16-ulp level and never less accurate than stock against the fp32 reference.

## 3. Kernel timing (verify_kernel.py, Part B): CONFIRMED at the kernel level

**Decode-step chain.** One CUDA graph holds the 80 layer GEMMs in model order, using the real distinct weights
of all 40 layers (1.28 GB, so every weight read is cold), plus the 3 lm_head calls of an MTP-2 step. Each
GEMM is preceded by its own per-token quant kernel, the same on both sides. Warmup is 5 replays; I report
the median of 50 replays, with stock and patched alternated (ABAB).

| case (layer M / lm_head M) | stock µs | patched µs | saved µs/step | layers / lm_head |
|---|---|---|---|---|
| c=1 (4 / 3,1,1) | 2214.4 | 1991.6 | **222.8** | 214.9 / 8.0 |
| c=4 as the author assumed (16 / 12,4,4) | 2223.6 | 2013.2 | 210.4 | 198.5 / 11.8 |
| M=32 (32 / 24,8,8) | 2224.0 | 2196.8 | 27.2 | 17.0 / 10.2 |
| c=16 (48 / 48,16,16) | 2348.7 | 2271.1 | **77.7** (claimed 130) | 82.5 / -4.8 |
| c=64 (192, fallback) | 2629.2 | 2619.9 | 9.4 (stdev 33, noise) | — |

Spot checks with the author's per-shape rotation method match or slightly beat their table:

| shape | stock µs | patched µs |
|---|---|---|
| gdn_qkvz M=4 | 20.0 | 15.2 |
| gdn_out M=4 | 8.4 | 6.4 |
| attn_qkv M=4 | 14.4 | 12.1 |
| gdn_qkvz M=48 | 21.2 | 18.4 |
| attn_qkv M=48 | 14.7 | 14.7 (fallback) |
| lm_head M=3 / 1 / 16 | 315.5 / 314.2 / 319.9 | 305.7 / 304.0 / 309.4 |

The skinny kernel wins clearly for M ≤ 16. The gain shrinks sharply for M 17-32, and at c=16 it is about
60 % of the claimed size.

## 4. In the real model (verify_insitu.py): the real numbers, and an integration hazard

I ran offline `LLM` with MTP-2, async scheduling, CUDA graphs and torch.compile, using stock attention files
(patches 01/02 do not touch these GEMMs). The patch files were mounted in both arms and only the flag
differed. nsys `--cuda-graph-trace=node` was enabled for a 48-token c=1 and c=4 window. Per-forward
dense-GEMM time comes from summing all 80 layer GEMM kernels of each forward pass:

| run | M ≤ 16 forwards (c=1, M=3) µs/forward | M 17-64 forwards (c=4, M=18) µs/forward | lm_head median µs |
|---|---|---|---|
| off1 | 1105.3 | 1151.9 | 315.9 |
| off2 | 1105.7 | 1153.5 | 316.3 |
| off3 | 1104.3 | 1151.2 | 315.9 |
| on2 (fresh compile) | 965.6 | 1108.2 | 312.2 |
| on3 (cache reuse of on2) | 965.3 | 1104.8 | 312.2 |

Per-kernel results in-situ (off → on):

| GEMM | c=1 bucket | c=4 bucket |
|---|---|---|
| gdn_qkvz | 20.22 → 18.13 µs | 21.06 → 19.63 µs |
| gdn_out / attn_o | 8.69 → 7.03 µs | 9.06 → 8.95 µs |
| attn_qkv | 15.26 → 14.37 µs | stays on stock (16.0 µs) |

**Real saving per decode step:** about 140 µs in the layers plus about 12 µs in lm_head, so **≈ 0.15 ms at
c=1** (claimed 0.19 ms). At c=4 it is about 45 µs plus about 10-20 µs, so **≈ 0.06 ms** (claimed 0.19 ms).

**Why c=4 is not M=16.** With MTP-2, vLLM rounds the CUDA-graph capture sizes to multiples of 3 for uniform
decode. The same `cudagraph_sizes.txt`-style list is used by serve.sh. c=1 therefore runs at M=3, and c=4
(12 tokens) is padded to M=18. The stock trace confirms this: it uses the 32x64 config. In that bucket
the patch gives about 1.4 µs on gdn_qkvz, about 0.1 µs on the out-projections and nothing on attn_qkv. The
author's c=4 estimate (and the M=16 rows of the table) do not match what the server actually runs. By the
same logic c=16 runs at M=48 (a multiple of 3), so the M=48 rows do apply there.

**Integration hazard (stale compile cache).** In my first "on" run (on1), nsys showed **all 80 layer GEMMs
still running CUTLASS**; only the eager lm_head used the skinny kernel. The log says why: "Directly load AOT
compilation from path ..." reused the graph compiled by the "off" run. vLLM's AOT cache key covers the vLLM
config, the known `VLLM_*` env vars and, at load time, the hashes of the traced source files. The new flag
`VLLM_SM120_SKINNY_FP8` is an unknown env var ("Unknown vLLM environment variable detected"), so it is not in
the key. Once `cutlass.py`, `sm120_skinny_fp8.py` and `sm120_skinny_fp8.cu` sit in the build tree, which is
the README's integration plan, turning the flag on or off reuses whichever graph was compiled first. If the
flag is turned on after a stock-flag compile, the layer path is silently disabled. on2 used a separate
`VLLM_CACHE_ROOT`, recompiled, and showed the skinny kernels.

The author's own e2e A/B is probably **not** affected: their off arm used the stock `cutlass.py`, so the
source-hash check at load forces a recompile. I could not prove this from their logs, which only grep for
"skinny". **Fix before integration:** register the op unconditionally and read the flag inside the op at
run time, so the compiled graph does not depend on the flag. Otherwise, document that the compile cache
must be cleared, or `VLLM_CACHE_ROOT` changed, whenever the flag is toggled.

**Wall-clock step time** (wall / spec-decode steps, which removes the MTP-acceptance confound; 3 rounds per
run):

| run | c=1 ms/step (rounds) | c=4 ms/step (rounds) |
|---|---|---|
| off1 | 6.79 (6.60-6.93) | 9.51 |
| off2 | 6.83 (6.80-6.84) | 9.53 |
| on1 (stale graph: lm_head only) | 6.74 | 9.39 |
| on2 | 6.60 (6.58-6.77) | 9.16 |
| on3 | **7.80 (7.64-7.81)** | 9.32 |
| off3 | 6.90 (6.71-6.91) | 9.32 |

Means over the clean runs: c=1 off 6.84 ms (off1-3) vs on 6.60 ms (on2 only), and c=4 off 9.45 ms vs on 9.16-9.32 ms. off3's dense-GEMM sums (1104.3 / 1151.2 µs per forward) match off1 and off2.

on3's GEMM kernels are identical to on2's (965 µs per forward), yet its c=1 step is 1.2 ms slower. The
wall-clock step time on this shared host is moved by things outside the GEMMs, probably CPU contention
from other agents. The on2 c=4 gain (-0.36 ms) is also larger than the GEMM saving can explain (0.06 ms).
**Wall-clock A/B at the 0.1-0.2 ms level is not reliable here**, and that applies to the author's ITL
numbers as well. The in-situ kernel sums above are the trustworthy measure.

## 5. Prefill claim (no change shipped)

I did not re-run the prefill sweep. Nothing is shipped for prefill, and the claim is a negative result:
config differences are inside ±3 % build and run noise, and the kernel is power-capped. I checked the author's
logs: the vLLM 128x128x128 coop kernel runs at about 690 TFLOPS at M=8192. With the patch, M > 64 goes
through the fallback, which I verified is bitwise equal to stock (M = 65, 96, 192 in Part A, and M = 1000 in
the author's test), so prefill numerics and speed are unchanged.

## 6. Summary of the claims

| claim | status |
|---|---|
| kernel correct, never less accurate than stock | confirmed |
| fallback is bitwise identical to stock | confirmed (`torch.equal`) |
| deterministic | confirmed |
| batch-invariant per row | **false across config boundaries** (stock is invariant) |
| 1-3 µs per layer call, 4-10 µs per lm_head call, M ≤ 16 | confirmed (in-situ: 0.9-2.1 µs per layer call, about 4 µs per lm_head call) |
| 0.19 ms per decode step at c=1 | **about 0.15 ms in-situ** (0.22 ms in my isolated chain) |
| 0.19 ms per decode step at c=4 | **about 0.06 ms**: c=4 runs at M=18, not 16 |
| 0.13 ms at c=16 | about 0.08 ms (isolated chain) |
| e2e c=1 ITL -0.11 to -0.17 ms | plausible, but inside host noise (my wall-clock runs vary by up to 1.2 ms between processes) |
| torch.compile / CUDA graph safe | yes when compiled fresh; **stale-compile-cache hazard when the flag is toggled** |
| prefill unchanged | confirmed through the fallback (bitwise) |

Other integration notes: the first start JIT-compiles with nvcc (about 1.5 min on this host; it needs nvcc in
the image and a writable `VLLM_CACHE_ROOT`). `_load()` sets `TORCH_CUDA_ARCH_LIST` process-wide as a side
effect; FlashInfer JIT does not read it. Kernel launches are not error-checked
(`C10_CUDA_KERNEL_LAUNCH_CHECK`).

## 7. Reproduce

The patched op is bind-mounted exactly as in `e2e_decode.sh`. The kernel test needs the GPU lock and takes
about 10 minutes:

```
K=kernels/dense-fp8-gemm; B=$PWD/$K/patch/b/vllm/model_executor/kernels/linear/scaled_mm
PKG=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/scaled_mm
flock <gpu.lock> sg docker -c "docker run --rm --gpus device=0 --ipc=host -u 1000:1000 -e HOME=/tmp \
  -e VLLM_SM120_SKINNY_FP8=1 -e VLLM_CACHE_ROOT=/tmp/vc -v $B/cutlass.py:$PKG/cutlass.py:ro \
  -v $B/sm120_skinny_fp8.py:$PKG/sm120_skinny_fp8.py:ro -v $B/sm120_skinny_fp8.cu:$PKG/sm120_skinny_fp8.cu:ro \
  -v /opt/intramind/hf-cache:/hf:ro -v $PWD:/work -w /work --entrypoint python3 vllm/vllm-openai:v0.32.0 \
  $K/verify_kernel.py AB"
```

In-situ: `verify_insitu.py <tag>` with the same mounts and `VLLM_SM120_SKINNY_FP8=0|1`, **a separate
`VLLM_CACHE_ROOT` per flag value**, `-e HF_HUB_CACHE=/hf/hub`, and optionally `NSYS_WINDOW=1` under
`nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node`. The runner I used is `run_insitu.sh`
in the scratchpad directory, and the per-forward analysis is `steps.py` / `kern.py` there.
