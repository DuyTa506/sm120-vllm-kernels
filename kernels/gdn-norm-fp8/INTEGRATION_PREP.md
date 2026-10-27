# Default-off packed GDN overlay

The default-off overlay passed the second owned stateful integration smoke:
all 30 ready layers, actual compiled QA→CUTLASS consumption, completed
piecewise/FULL graph capture, four greedy decode scenarios and ten finite
NLL requests. See `results/INTEGRATION_GPU.md` and its source-hashed JSON.
This establishes execution, not paired quality/state equivalence or a speedup.
The first capture-size failure and split-config confound remain preserved in
`CAPTURE_WARMUP_FIX.md` and `SPLIT_CONFIG_FIX.md` under results.

The original MTP2 serving FX contains 30 calls to
`qwen_gdn_attention_core_fused_norm_packed`. That op calls the existing fused
MTP decode implementation for eligible pure decode, or `_forward_core` plus
`_rms_norm_gated_cuda` for prefill/mixed batches. The latter directly invokes
FLA, independent of `RMSNormGated._forward_method`. The BF16 result then feeds
compiled native QuantFP8 and the CUTLASS output projection. Saved evidence:
`results/qa_interface_review_cpu.json`.

The overlay adds a constant enabled/disabled branch inside the existing
packed route. The enabled branch allocates core/Q/S buffers and calls
`gdn_attention_core_norm_fp8_packed_experimental`, then passes the existing
`QuantizedActivation` wrapper into normal `self.out_proj`.
There is no symbolic token-count branch in model Python. vLLM traces one
dynamic graph and later uses a guard-free wrapper; a Python `T>=1024` branch
would constrain that graph. Inductor compile ranges do not repair it.

Inside the opaque op, actual metadata selects behavior:

- No metadata: warm the original prefill kernels and write zero FP8 plus
  scale floor `1/(448*512)`, matching quantization of baseline's zero core.
- At least 1024 actual tokens and at least one prefill: delegate to the
  original `_forward_core` with contiguous B/A, then run the frozen explicit
  norm/FP8 kernel on actual rows. Preserve strided Z. Padded rows receive
  zero Q and scale floor, without reading arbitrary padded Z values.
- Decode of any size, zero actual tokens, or smaller prefill: delegate to
  original `_forward_core_fused_norm_packed`, then call the validated compiled
  native QuantFP8-to-output wrapper on **all padded rows**. It writes Q/S
  directly into caller-owned buffers. This retains original recurrent state/checkpoint handling and
  fused MTP decode selection. The outer quantization boundary changes.

The fallback now uses the GPU-validated `fallback_quant_out.py` wrapper,
which calls the installed raw QuantFP8 method and both destination copies
inside one `torch.compile(..., dynamic=True, fullgraph=True)` graph. Its two
generated specializations each have one fused kernel writing caller-owned
Q/S, with no intermediate Q/S allocations or separate copy launches. The
original quantizer remains unchanged for the oracle and ordinary Tensor path.
Caller-owned buffers retain `eager_break_during_capture` address stability.

The wrapper passed 60 numerical and 36 changed-input replay checks, including
unchanged real output GEMM. Its small-shape graph timings are lower than
native quantization **plus two separate copies**; native-only quantization
was not timed as a comparator. Decode regression versus original serving
remains an open integration measurement. See
[`results/FALLBACK_QUANT_OUT_GPU.md`](results/FALLBACK_QUANT_OUT_GPU.md).

The consumer adapter is limited to this GDN's original exact CUTLASS FP8
kernel instance. It copies its original five fields, including the original
QuantFP8 object, and inherits `apply_weights` and `apply_scaled_mm` unchanged.
Its post-load override first calls the original weight processor, then
validates loaded FP8 B `[4096,2048]` with stride `(1,4096)`, FP32 channel
scales, BF16 norm weight `[128]`, logical output width, and absent activation
scale/upper-bound/padding. Only then does it advertise dynamic-token QA and
enable the producer. Ordinary Tensor input still uses the original quantizer.
No generic linear forward, dense quantization or GEMM implementation changes.

Construction guards require SM120, TP/PP1, compiled Inductor mode3, disabled
CUDA QuantFP8 custom op, existing packed GDN route, non-interleaved layout,
exact norm/shape/FP8 scheme/kernel/dtype contract, no LoRA, no bias or
unquantized branch, and the supported piecewise split configuration. The
original compiled QuantFP8 callable is verified by its wrapped raw method.
Unsupported configurations leave the original producer path enabled.

Post-load startup constructs and warms the validated output wrapper for T1,
T2, maximum scheduled T, every configured graph bucket, and each derived
uniform-decode bucket. MRV2 rounds configured sizes up to the fixed MTP
query length, bounded by max sequences and maximum capture size. The stored
config list stays unchanged, so warming only that list is insufficient.
The candidate requires fixed MTP and disables itself for dynamic speculative
schedules or DBO, whose additional shape laws are outside this experiment. It also warms the explicit kernel's
prefill layout with the actual X/Z strides. It refuses startup under stream
capture and rejects unwarmed graph buckets in the runtime fallback with the
actual padded T, actual-token count, prefill count and complete warmup list. Warming
is startup GPU work when the overlay is eventually enabled, not work run by
the CPU preparation. The recorded second startup completed the tested graph captures with this
warmup plan. It does not prove every unsupported or future shape is safe.

The wrapper file is copied byte-for-byte into `gdn_norm_fp8_quant_out.py` in
the overlay; its frozen docstring predates GPU validation. CPU tests verify
byte identity and the exact compilation arguments. No backend/options
overrides are supplied beyond the tested dynamic/fullgraph settings.

The new custom op mutates caller-owned core/Q/S, has a fake implementation,
and uses `eager_break_during_capture`. Its split name must be supplied **before
VllmConfig construction**; the helper never mutates a hashed config later.
`integration_config.py` reads the installed source with AST, preserves the
provided base JSON and all 19 completed baseline split ops (17 attention
ops plus two KV-update ops), and adds the new name. It explicitly preserves
`pass_config.fuse_rope_kvcache_cat_mla=False`. The first generator skipped
default completion and changed this pass; the source-derived correction and
regression are recorded in `results/SPLIT_CONFIG_FIX.md`. Unsupported explicit
base split/pass configurations are rejected. `results/integration_compilation_config.json` records that source hash
and the emitted default merge.

```sh
# CPU only, in the pinned container; pass the existing compilation JSON if any.
CUDA_VISIBLE_DEVICES= python3 /work/kernels/gdn-norm-fp8/integration_config.py \
  --base-json '{}'
CUDA_VISIBLE_DEVICES= python3 /work/kernels/gdn-norm-fp8/test_integration_guards.py
CUDA_VISIBLE_DEVICES= python3 /work/kernels/gdn-norm-fp8/verify_fallback_quant_out.py --cpu-preflight
```

To stage a candidate after the active GPU queue ends, mount the five files
listed in `overlay_manifest.json` at their corresponding vLLM package paths,
set `VLLM_GDN_NORM_FP8_FUSION=1` in the owned worker environment, and pass the
emitted JSON via `--compilation-config`. Use a distinct compilation cache and
retain the selected MoE stack/configuration for the incremental comparison.
The flag defaults to zero. This note does not modify root's server lifecycle.

CPU tests cover exact installed split/pass completion, MRV2 descriptor coverage, the initial missing
186-token case, request/capture bounds, diagnostic guard failures, all
configuration rejections, dispatch thresholds, large
pure-decode fallback, metadata-free zeros, padded tails with NaN padded Z,
original-core delegation, strided Z offset 8192, mutating fake schema and one
dynamic CPU Dynamo graph over T3/31/1025. They do not execute the CUDA op.

Standalone quant-out repeat command (the recorded GPU run passed):

```sh
TORCH_LOGS=output_code python3 /work/kernels/gdn-norm-fp8/verify_fallback_quant_out.py \
  --run-gpu --bench --tokens 1 2 3 31 64 192 1025 \
  --snapshot /root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/1c3f884bc99aac2524f6d49bcbac8c88401afd66
```

This checks FP8 bytes, scale bits, unchanged real layer18 GEMM and changed-input
CUDA graph replays before timings. It retains reference temporaries during
graph capture. Keep generated output code and selected launch configurations;
identical native quant source has selected different Inductor block/warp
configurations in previous runs. CPU FX evidence does not predict the GPU
configuration or prove elimination of separate copies.

The scoped integration has established all 30 ready layers, the new-op/QA
boundary in compiled FX, metadata-free startup, configured graph capture,
and actual decode/scored-prefill execution. Before adoption, complete matched
greedy/NLL and seven-task serving comparisons, measure decode regressions,
and validate any required mixed-batch accepted-state/checkpoint parity and
integrated padded Q/S exactness. Agentic evaluation belongs only to the final
selected combined configuration. No end-to-end improvement is claimed here.
