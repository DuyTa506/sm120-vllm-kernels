# GDN norm + SiLU + token FP8 experiment

The explicit-layout candidate passes the recorded standalone exact checks;
the original automatic-layout candidate fails. The default-off overlay now
also completes the full scalar + GDN serving comparison: all 17 tasks, four
greedy decode cases, 30 cold NLL calls and 20/20 needles, with all 30 layers
ready and fixed FlashInfer tables at startup/shutdown.

Measured effects are mixed. Against scalar, 32k TTFT improves 7.6% and long-doc
throughput improves 9.1%, while 8k TTFT worsens 7.4% and greedy c=64 throughput
falls 8.71%. The complete stack beats `next` on all six cold-prefill medians,
but loses 8.17% at greedy c=64 and retains a licenses-30000 NLL flag. GDN is
not selected for the final stack yet; a focused reversed-order GDN→scalar
repeat precedes selection. Agentic runs only on the final selected stack versus
`next`. See [`results/COMPOSED_SERVING.md`](results/COMPOSED_SERVING.md).
The earlier scoped integration smoke remains separately recorded in
[`results/INTEGRATION_GPU.md`](results/INTEGRATION_GPU.md).

A separate layer18 state probe now passes actual eager long-prefill,
mixed speculative decode+prefill and pure-decode comparisons. All six T/T+17
variants have exact Q/S and updated private convolution/recurrent cache bytes;
padding checks pass and live caches remain untouched by the probe. Checkpoint
and mixed non-speculative metadata were not observed. This is separate from
paired model quality or serving performance; see
[`results/STATE_PROBE_GPU.md`](results/STATE_PROBE_GPU.md).

The first integration failed on a derived MTP capture size, and its generator
missed two baseline KV-update splits. Both failures and their corrections are
preserved in [`results/CAPTURE_WARMUP_FIX.md`](results/CAPTURE_WARMUP_FIX.md) and
[`results/SPLIT_CONFIG_FIX.md`](results/SPLIT_CONFIG_FIX.md).

The GPU run ledger is [`results/EXPLICIT_GPU.md`](results/EXPLICIT_GPU.md).
Recorded results are summarized in
[`results/explicit_validation_summary.json`](results/explicit_validation_summary.json):

- Initial explicit run: 144 numerical cases and 72 changed-input graph replays.
- Real norm/output-projection weights from layers 0/18/38: 648 numerical
  cases, 324 replays and 432 timing records.
- Actual 8192-token prefill capture from those layers: 3 captured BF16 norm
  checks, 9 exact FP8/scale/unchanged-GEMM checks and 27 replays, all passing.

Exact means zero FP8-byte, scale-bit and checked GEMM-output differences.
The oracle is installed FLA norm followed by compiled native QuantFP8.
Saved server FX confirms the default packed GDN path calls FLA directly
inside its opaque core, independent of the norm module's dispatch setting.
See [`results/qa_interface_review_cpu.json`](results/qa_interface_review_cpu.json).
The earlier `_output_projection` hook proposal would miss this packed path.

Isolated CUDA-graph timings for layer 18, eight warps, including the
baseline gate copy and the same real output GEMM (median of recorded AB/BA
medians, microseconds):

| Tokens | Reference chain | Explicit fused chain |
| ---: | ---: | ---: |
| 4096 | 173.09 | 146.82 |
| 8192 | 391.05 | 286.90 |
| 16384 | 925.94 | 584.51 |

These are microbenchmarks. Both chains retain their captured output buffers.
The separate full serving comparison above measures mixed end-to-end effects;
it does not assign those effects entirely to this isolated kernel. The scoped
layer18 state/padding check passes; its unobserved paths remain explicit.
Cold NLL diagnostics do not establish model-quality equivalence.

The original failure and reduction-order diagnosis are preserved in
[`DIAGNOSIS.md`](DIAGNOSIS.md). The successful capture is
`results/activations_8k_v2`; the first capture's RPC serialization failure
and owned-process cleanup are recorded separately. See [`CAPTURE.md`](CAPTURE.md).
The compiled-output fallback passes its separate exactness checks;
[`results/FALLBACK_QUANT_OUT_GPU.md`](results/FALLBACK_QUANT_OUT_GPU.md) records
its generated code and limited timing comparison.
The default-off overlay and required split configuration are described in
[`INTEGRATION_PREP.md`](INTEGRATION_PREP.md). The candidate was mounted only
in owned experiment servers; it remains opt-in and default-off.

`fused_explicit.py` implements one Gluon CTA per token with the FLA reduction layout.
The frozen source docstring predates its GPU validation; the recorded results
above are authoritative. `fused.py` remains the preserved failing candidate. Inputs are BF16 X/Z
`[T,32,128]`, BF16 weight `[128]`, epsilon 1e-6, no bias. Z may have the actual
QKVZ slice strides `(12288,128,1)`. Norm reduces each head; SiLU gating runs
in FP32; the result explicitly rounds through BF16 before one dynamic FP8
scale over all 4096 token channels. Outputs are contiguous E4M3FN `[T,4096]`
and FP32 `[T,1]`. Zero rows use scale `1/(448*512)`.

`verify_bench.py` does not import torch/vLLM or initialize CUDA unless
`--run-gpu` is supplied. It uses installed FLA RMSNormGated and compiled
`QuantFP8.forward_native` as the serving reference; it separately reports
differences against installed `ops.scaled_fp8_quant`. Exact scale bits and
FP8 bytes are required against the compiled reference. It stops on mismatch
before timing. Optional checkpoint loading tests real norm weights and the
unchanged CUTLASS output projection. The capture checks cover one eager request
and selected layers. The separately measured serving/quality diagnostics above
retain their own scope; broad model accuracy is not established.

Run these only after the GPU is available, inside the vLLM 0.32.0 environment
with this directory mounted:

```sh
# CPU-safe status and CLI checks:
python3 verify_bench.py
python3 verify_bench.py --help

# Numerical/graph validation, then timing, with synthetic weights:
python3 verify_bench.py --run-gpu --mode verify --implementation explicit --warps 4 8 16
python3 verify_bench.py --run-gpu --mode bench --implementation explicit --warps 4 8 16

# Real norm and output-projection weights from the mounted local snapshot:
python3 verify_bench.py --run-gpu --mode both --implementation explicit --with-gemm --warps 8 \
  --snapshot /root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/1c3f884bc99aac2524f6d49bcbac8c88401afd66
```

The verifier includes odd token counts, strided and contiguous gates, zeros,
small scale-floor cases, extreme positive/negative gates, head outliers,
rounding-edge magnitudes and changed-input graph replays. Default checkpoint
layers are 0, 18 and 38. JSONL records include every mismatch count, both
reference paths, CUDA graph timings in A/B and B/A order, and separate
contiguous-gate versus gate-copy-inclusive baselines. Three distinct input
buffers are used per timing graph. With `--with-gemm`, each chain is also
timed with the same real output GEMM. GPU event timings do not include Python
allocation overhead; CUDA graphs hold baseline temporaries in their pool.

The verifier defaults to the original candidate for provenance; pass
`--implementation explicit` for the validated candidate. The explicit kernel
and verifier remain frozen at their tested hashes. GPU steps require an
exclusive scheduled slot, writable runtime caches and the local snapshot.

## Offline compilation

`compile_offline.py` compiles the unchanged kernel for an explicit SM120
target without initializing CUDA or launching a kernel. Artifacts and tool
versions are in [`results/offline_sm120/SUMMARY.md`](results/offline_sm120/SUMMARY.md).

| warps | registers/thread | shared bytes | spill loads/stores |
|---:|---:|---:|---:|
| 4 | 64 | 256 | 0/0 |
| 8 | 43 | 32 | 0/0 |
| 16 | 38 | 64 | 0/0 |

The generated code retains FP32→BF16→FP32 rounding before quantization and
the E4M3 conversion. The per-head reduction precedes the token-wide maximum.
Compilation also fuses some multiply/add operations; the original candidate subsequently failed the recorded GPU comparison.
The separate explicit-layout offline artifacts are under
`results/explicit_layout_offline/`; all six compiled specializations have zero spills. These
resource counts are preparation evidence, not a speed measurement.
