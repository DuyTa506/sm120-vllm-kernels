# Scalar FC1-scale dispatch experiment

Status: prepared and CPU-preflight checked; **GPU compilation, correctness,
CUDA-graph replay, and performance validation are pending**. No GPU work was
run for this experiment. Neither the `shared/` backend nor the existing
workspace verifier/harness nor installed container sources were changed.

## Change and limits

`scalar/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py`
is a separate copy of the shared-workspace candidate. Set
`VLLM_B12X_SCALAR_INPUT_SCALE=1` **before importing this overlay** to enable
its single added weight-loading hook. `VLLM_B12X_FIX_SCALES=1` is also required.
The existing shared-workspace flag is independent and is enabled in the
comparison script for both variants.

The hook replaces an exactly uniform, finite, positive FC1 input-scale
vector `[256]` with an independent `[1]` tensor. All nonuniform values remain
unchanged, including their original tensor identity. It is restricted to
BF16, SiLU, E=256, H=2048, I=512, top-k=8, local E=global E, no EP, and the
existing hybrid threshold >=1024 tokens. Both weight alphas and FC2 scales
retain their existing representation. Small batches keep the CUTLASS path.

Uniformity is checked with a 1 KiB CPU copy once per eligible layer during
weight preparation. This is a **load-time synchronization**, not a forward
or CUDA-graph replay synchronization. `apply`, wrapper ownership, and CUTLASS
execution are identical to `shared/`; the CPU preflight compares their ASTs.

## Installed-source findings

Sources were read from `qwen-prefill-dev` at
`/usr/local/lib/python3.12/dist-packages/flashinfer/` on 2026-10-27.
Line numbers below refer to this installed copy, not upstream latest.

- `fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py:2437` remembers
  `input_gs.numel() == 1` before expansion and passes the result to
  `_get_dynamic_kernel` at line 2458. The compiler gates this on FP4 and
  top-k<=32 at lines 2192-2197; the shared bit participates in its cache key.
- `.../moe_dynamic_kernel.py` selects `MoEGatedDynamicKernel` for gated
  activations, SF vector=16, tile=(128,128), H<=16384, I<=512, and top-k<=16
  for the nonshared path (<=32 shared). Qwen's H2048/I512/top-k8/SiLU matches
  **both** branches, provided the workspace tile M is 128. The verifier checks
  that actual workspace condition. Default NVFP4 dynamic selection starts
  above 1024 routed rows (128 tokens at top-k8); this experiment's existing
  vLLM hybrid threshold remains 1024 **tokens**, well inside that band.
- `.../_moe_dynamic/gated.py:3448` already compares selected route scales and
  quantizes each 16-value input block once when they are equal. Scalar dispatch
  does **not** remove eight quantizations from the current uniform baseline.
  It hoists scale loading, skips per-block equality checks, caches route
  addresses, and unrolls top-k8 stores (lines 3122, 3232-3303). It still writes
  each route's packed data and block scales.
- `fused_moe/cute_dsl/b12x_moe.py:665-679` broadcasts
  `w1_alpha[E] * input_global_scale[1]` into the same `[E]` folded alpha as
  `w1_alpha[E] * uniform_input_global_scale[E]`, caching by tensor pointers
  and versions. `launch_sm120_moe` skips its own fold when wrapper weight
  views are supplied (dispatch lines 3479-3486). The overlay does not fold
  alpha itself, avoiding double scaling. FC2 remains `w2_s2 * a2`.
- `flashinfer/cute_dsl/fp4_common.py:2640-2649` uses
  `block_scale = amax / (6 * global_scale)` and
  `payload = x / (quantized_block_scale * global_scale)`. This is the existing
  dequant-style a13 convention; scalar versus repeated vector changes no value.
- Direct-micro's own scale contract is multiplier form. Dispatch lines
  1627-1644 can invert reciprocal scales, but only when
  `input_scales_are_reciprocal=True`. `B12xMoEWrapper.run` does not pass that
  argument, leaving the launch default False. The existence of this bridge
  does **not** establish safety for sending this backend's static scales to
  direct-micro. The experiment preserves the conservative hybrid gate.
- EP remains unsupported: the wrapper rejects local E != global E, and the
  dynamic kernel indexes routing state by global expert IDs. Scalar dispatch
  is not an EP-enablement change.

**Cost to measure:** `_expand_to_experts` (dispatch line 1494) performs
`scalar.expand(E).contiguous()` on every launch. The scalar case therefore
adds a small device broadcast/copy and an eager temporary allocation; CUDA
graph capture retains its allocation and replays the copy. Folded alphas are
cached, but this expansion is not. Any kernel savings compete with this cost.
No speedup or net allocation improvement is claimed. Avoid changing FlashInfer
internals until the simple full-call experiment provides evidence.

Source SHA256 values:

| File under FlashInfer | SHA256 |
| --- | --- |
| `fused_moe/cute_dsl/b12x_moe.py` | `c09065511e93c0ccf40e4d99f9215157aac9c26e0796ab41d1c52d396b45a2ff` |
| `fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py` | `c518e65d6bfd7f08db1e5261e20795fd020e82e681699729171bc2fd5331239a` |
| `fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py` | `7532e1338df8e80c4b8f0bd6473c2ffeda9a5028018c77775c4e9fe2fa47b072` |
| `fused_moe/cute_dsl/blackwell_sm12x/_moe_dynamic/gated.py` | `993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445` |

## Validation and later run

`verify_scalar_dispatch.py --preflight` runs the actual hook extracted from
its AST on CPU tensors. It checks 24 uniform/invalid/nonuniform/configuration/
flag cases, exact scalar/vector alpha-fold equivalence, and unchanged forward
methods and existing scale preparation. The initial preflight failed with
`scalar selection not implemented`; after implementation all 24 passed with
`cuda_initialized=false`. Both Python files passed source compilation.

Repeat the CPU-only check:

```bash
sg docker -c 'docker exec -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 qwen-prefill-dev python3 /work/kernels/moe-nvfp4/verify_scalar_dispatch.py --preflight'
```

Only after Arm A releases the GPU, run the explicit GPU mode:

```bash
sg docker -c 'docker exec -e PYTHONDONTWRITEBYTECODE=1 qwen-prefill-dev python3 /work/kernels/moe-nvfp4/verify_scalar_dispatch.py --gpu --tokens 1023 1024 8192 --layers 0 19'
```

This loads each overlay directly from `/work` without mounting over installed
sources. It uses real Qwen weights/hidden states, checks actual dynamic shared
flags during warmup/capture, verifies the gated dispatch geometry and alpha
shape, and compares captured full `apply()` calls (including scale expansion,
ID masking, and output copy). It interleaves timing in both orders and checks
real/hot/masked/shifted routing over replay. A deliberately nonuniform FC1-scale
case confirms the generic path remains selected. That nonuniform case is a
representation-preservation comparison against `shared/`, not a new checkpoint
accuracy claim. Tolerance is relative L2 <= max(0.002, 3 * baseline-repeat noise).
Independent source review found no substantive defect. Its coverage finding
was addressed: the nonuniform eight-hot fixture selects expert 17 explicitly
alongside experts 0–6, guaranteeing mixed selected scales are exercised.
Negative-ID tests run only in the b12x band; the CUTLASS boundary case tests its
unchanged route contract. GPU validation must pass and show a repeatable full
call benefit before adoption. No benchmark results exist yet.
