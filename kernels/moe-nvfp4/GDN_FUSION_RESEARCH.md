# GDN norm + gate + token FP8 quantization experiment

Read-only investigation, 2026-10-27. No GPU jobs, kernel changes, or performance
claims. Target: `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast`, TP=1, SM120, installed
vLLM 0.32.0 in `qwen-prefill-dev`. The smallest useful experiment replaces the
GDN output RMSNormGated and its following dynamic per-token FP8 quantizer,
keeping the FP8 output projection GEMM unchanged.

## Exact path and shapes

Paths below are relative to `/usr/local/lib/python3.12/dist-packages/vllm/`
inside that container:

| File / function | Relevant behavior |
|---|---|
| `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:932`, `_output_projection` | `self.norm(core_attn_out, z)` followed by `self.out_proj(...flatten(-2))`. |
| Same file, `forward_cuda` | Allocates BF16 core output `[T,32,128]`; splits BF16 QKVZ `[T,12288]` into QKV and gate. Gate view has strides `(12288,128,1)` and starts at column 8192. |
| `model_executor/layers/layernorm.py`, `RMSNormGated.forward_cuda` | Calls **`third_party/flash_linear_attention/ops/layernorm_guard.py`**, not the similarly named Mamba ops file. |
| `third_party/flash_linear_attention/ops/layernorm_guard.py`, `rmsnorm_fn`, `_layer_norm_fn_impl`, `layer_norm_fwd_kernel` | Norm over 128 values per head, FP32 arithmetic, SiLU after norm, BF16 output. Also writes unused FP32 rstd `[32*T]`. |
| `third_party/flash_linear_attention/ops/utils.py`, `input_guard` | Calls `.contiguous()` on all tensor arguments, including the strided gate. A direct strided-input candidate can avoid this gate materialization too. Confirm the copy in the compiled execution graph before attributing time to it. |
| `model_executor/layers/quantization/input_quant_fp8.py`, `QuantFP8.forward_native` | Current compiled quantization: one dynamic FP32 scale per **4096-element token**, not per head; FP8 E4M3FN output `[T,4096]`, scale `[T,1]`. |
| `model_executor/kernels/linear/scaled_mm/ScaledMMLinearKernel.py`, `FP8ScaledMMLinearKernel.apply_weights` | Obtains quantized input/scales, then calls `apply_scaled_mm`. |
| `model_executor/kernels/linear/scaled_mm/cutlass.py`, `CutlassFP8ScaledMMLinearKernel.apply_scaled_mm` | Calls `ops.cutlass_scaled_mm` with transposed FP8 weight `[4096,2048]`, channel weight scales, BF16 output `[T,2048]`. |

Checkpoint snapshot `1c3f884bc99aac2524f6d49bcbac8c88401afd66/config.json`
confirms 40 layers, 30 `linear_attention`, 10 full attention, 32 value heads,
128 values/head, hidden size 2048, epsilon `1e-6`, BF16 model dtype, dynamic
token activation FP8 and channelwise weight FP8 for GDN `out_proj`.
Cached baseline norm IR confirms X, Z, W and Y are BF16. Weight is `[128]`,
shared across heads; it is the stored weight, without a Gemma-style `+1`.

The installed `_output_projection` docstring says fusion is eligible, but CUDA
`compilation/passes/fusion/rms_quant_fusion.py` matches ordinary RMSNorm and
fused-add RMSNorm. Gated support in `rocm_aiter_fusion.py` is
`AiterRMSNormGatedFp8GroupQuantPattern`, restricted to ROCm AITER and group 128
scales `[T,32]`. That replacement does not implement this CUDA/token-scale
contract.

## Numerical contract to preserve

For token `t`, head `h`, channel `d`, all arithmetic below is FP32 except the
explicit conversions:

```text
r[t,h] = rsqrt(sum_d(float(x[t,h,d])**2) / 128 + 1e-6)
v[t,h,d] = ((float(x[t,h,d]) * r[t,h]) * float(w[d]))
           * (float(z[t,h,d]) * sigmoid(float(z[t,h,d])))
y[t,h,d] = float(round_to_bf16(v[t,h,d]))
a[t] = max_(h,d)(abs(y[t,h,d]))
s[t] = max(a[t] * float32(1/448), float32(1/(448*512)))
q[t,h,d] = round_to_e4m3fn(clamp(y[t,h,d] * reciprocal(s[t]), -448, 448))
```

The BF16 round-trip before both `amax` and scaling is mandatory: eliminating
the materialized tensor must not eliminate its rounding. Do not use raw FP32
`v` to calculate the scale, apply BF16 gating early, normalize all 4096
channels together, or substitute 32 head scales. Preserve multiply order,
128-wide reduction layout, `tl.rsqrt`, `tl.sigmoid`, reciprocal behavior and
FP8 rounding; layout/reassociation changes can still change outputs despite
matching this algebra. Check generated IR for the lossy BF16 conversion.

Concrete compiled-source evidence is in the current baseline cache:

```text
.cache/vllm/qwen-prefill-cutlass/torch_compile_cache/torch_aot_compile/
683d4d9db6ec721f3c5261735c6a6cf09da19fb25d46b32da5273fe333f44eae/
inductor_cache/e6/ce6mt6ek3o6uhf4bth2moedw3qrv546ztkw2wxgiqgiyq3l64scq.py
```

Its generated `triton_red_fused__to_copy_abs_amax_clamp_cutlass_scaled_mm_div_mul_reciprocal_unsqueeze_view_0`
reads BF16 rows of 4096, multiplies amax by `0.002232142857142857`, clamps
scale to `4.359654017857143e-06`, then reciprocal-multiplies, clamps to ±448,
and casts to `tl.float8e4nv`. This generated variant has separate amax and
quantization read loops. Eager `QuantFP8.forward_cuda` instead calls
`ops.scaled_fp8_quant`; test it separately, but use the compiled serving path
as the decisive numerical and timing reference. Initial scope assumes
`scale_ub=None`, no padding, no bias, finite BF16 inputs.

## Trace normalization and honest bounds

`bench/nsys_kernels.py` divides both summed duration and call count by `NREQ`
(default 4). `scripts/nsys_capture.sh` captures four 8k requests after warmup.
Thus `results/nsys/kernels_8k_tuned.txt` reports **60 calls per request**, not
60 calls over the capture:

| Old tuned trace | ms/request | calls/request |
|---|---:|---:|
| `layer_norm_fwd_kernel` | 1.09 | 60 |
| GDN token FP8 quant kernel above | 0.58 | 60 |
| Sum | 1.67 | 120 |

The model has 30 GDN layers. The original prefix-cache checkpoint path splits
8192 into two model forwards, 4256 + 3936 (root README, flag
`VLLM_GDN_INTERNAL_CKPT`); 30 layers × 2 explains the 60 calls. CONFIG=next
uses internal checkpoints and needs 30 calls for this single-chunk 8k case.

Current four-request SQLite captures independently confirm 120 total norm
launches each, grid `(65536,1)`, block `(32,1)`; four heads/program corresponds
to `8192*32/4 = 65536` programs:

| Current saved trace | norm calls/request | norm ms/request |
|---|---:|---:|
| `results/nsys/raw/shared_a2_8k.sqlite` | 30 | 2.32903725 |
| `results/nsys/raw/shared_b1_8k.sqlite` | 30 | 2.97741975 |

These ran alongside training. Their plain CUDA graph tracing hides the
compiled quant kernels, so neither provides a current norm+quant bound or
an isolated performance comparison. The old 1.67 ms is only an optimistic
upper bound on removable **summed kernel execution time of those two old
operations**, assuming a zero-cost replacement. It is not a promised TTFT
saving, a current-path bound, or a bound that includes gate copies. Real
fusion still performs normalization, gating, cross-head amax and FP8 writes.
Do not divide 1.67 by two to predict current time: the token work is unchanged
while launch count and tile shapes differ. Re-profile with graph-node tracing
after training before attaching a speed claim.

## Traffic estimate, excluding GEMM

Let `E=T*4096`, ignoring cached weight traffic and tiny scale traffic.
The norm reads X/Z (4E bytes) and writes BF16 Y (2E); a one-read quantizer
reads Y (2E) and writes FP8 Q (E): **9E bytes**. A fused kernel that retains
values on chip reads X/Z and writes Q: **5E bytes**. Removing Y materialization
saves **4E bytes**, or **128 MiB/layer at T=8192**, **3.75 GiB across 30 layers**;
16k doubles those numbers. Removing rstd adds 1 MiB/layer at 8k.

The inspected quant source reads Y twice, making its source-level pair 11E
and potential instruction-level removal 6E (192 MiB/layer at 8k). Gate
contiguous-copy elimination could remove another 4E (128 MiB/layer), separately.
These are tensor-access counts, not measured DRAM traffic: L2 reuse, compiler
load reuse, transactions, and spills determine actual memory traffic. Do not
convert them directly into latency using advertised memory bandwidth.

## Minimal implementation and acceptance plan

1. Prototype only `gdn_norm_gate_fp8(x,z,w) -> (q,s)` in a new standalone
   experiment file. Fixed head shape `[32,128]`, one CTA/program per token,
   start with 8 warps and compare 4/16 only after correctness. Reduce each
   128-wide head, apply the gate, explicitly round through BF16 in registers,
   reduce head maxima to one token maximum, then write FP8 and FP32 scale.
   Accept actual X/Z strides; avoid flattening Z through a copying reshape.
   No global intermediate Y/rstd and no change to GEMM or model weights.
   Inspect registers/spills and reduction layout; extra cross-warp reduction
   and register pressure could outweigh traffic/launch savings.
2. Compare first against the installed norm plus compiled
   `QuantFP8.forward_native` on identical inputs. Cover T = 1, 3, 31, 128,
   1023, 1024, 1025, 4096, 8192, 16384; contiguous Z and the actual QKVZ slice;
   zeros, small magnitudes around the scale floor, ±large gates, outliers in
   different heads, and values near BF16/FP8 rounding ties. Use real norm
   weights and saved activations from representative early/middle/late layers
   when GPU execution is available. Compare debug BF16 Y, exact FP32 scales,
   FP8 bytes, then BF16 output from the unchanged CUTLASS GEMM. Require exact
   agreement initially; investigate mismatches before choosing any tolerance.
3. Benchmark the full baseline pair and fused kernel, then both with the same
   output GEMM. Include gate materialization in both realistic measurements
   and report a contiguous-Z control. Use CUDA events and graph replay, both
   A/B orders, multiple input buffers, preallocated outputs, warmup and
   several repeats. Time prefill 4k/8k/16k separately from tiny rows. Record
   launches, allocations and spills; ensure the entire fused tensor stays
   live through amax without off-chip spills.
4. Only after a useful microbenchmark result, add an opt-in serving experiment
   at `_output_projection`, guarded to CUDA SM120, TP=1, BF16, `[32,128]`,
   SiLU, no bias, epsilon 1e-6 and dynamic-token/channel FP8 CUTLASS. A direct
   branch can call the verified fused op and the selected backend's
   `apply_scaled_mm` with existing weights/scales, retaining BF16 output and
   normal return behavior. Keep unsupported/small-token paths on baseline.
   **Do not pass raw FP8 through `self.out_proj`:** it will quantize again;
   `CutlassFP8ScaledMMLinearKernel.input_quant_key()` currently accepts
   external activations only for static tensor quantization. Do not widen
   that public contract merely for the prototype. Guard/fall back around
   the separate `enable_fused_gdn_spec_decode` branch, which already bypasses
   `_output_projection` and must not receive a second normalization.
5. For a later general integration, extend CUDA `RMSNormQuantFusionPass` with
   `MatcherRMSNormGated` + flatten + `MatcherQuantFP8(kFp8DynamicTokenSym)`;
   preserve the two outputs and unchanged CUTLASS consumer. Verify actual
   matches for both custom/native norm forms and strided gate copies; the
   existing ROCm pattern is a structural reference, not a compatible op.
   Confirm 30 norm/quant pairs become 30 fused calls in a clean 8k graph-node
   trace. Run baseline/candidate NLL/logprob checks, deterministic token
   comparisons, prefill TTFT at 8k/16k and decode regression checks with
   disjoint warmup seeds and controlled prefix-cache state. Accept only a
   repeatable whole-path improvement after training stops.
