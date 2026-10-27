# Explicit-layout GDN fusion GPU evidence

The explicit-layout standalone kernel passes the executed numerical tests.
Serving integration remains unvalidated and unmounted. This report does not
claim a full-model speedup.

## Numerical checks

- Initial explicit run: tokens3/16/31/1025, warps4/8/16, both contiguous and
  strided gate inputs, synthetic edge cases and changed-input CUDA graph replay.
- Real-weight run: **648 numerical cases and324 graph replays**, layers0/18/38,
  tokens3/16/31/1024/8192/16384, all three warp counts. FP8 bytes, FP32 scale bits
  and unchanged real output-projection GEMM BF16 bits match the reference.
- Actual eager 8192-token capture v2: **9 captured-input comparisons and27 graph
  replays**, all passing for those layers/warp counts. Captured norm BF16 outputs
  match the FLA reference, then fused FP8/scale and GEMM outputs match exactly.
  This is actual eager input coverage, not an end-to-end compiled-serving test.

The reference is FLA gated RMSNorm followed by compiled-native QuantFP8.
Independent inspection of next FX/Inductor and A3/B2 traces confirms this chain
at the packed GDN core's prefill fallback. The module's initial native norm
method does not describe this opaque runtime path. Each historical four-request
8k trace contains120 FLA norm calls. See `qa_interface_review_cpu.json`.
CUDA eager FP8 quantization is a different numerical contract and is reported
separately by the verifier; it is not silently treated as equal.

## Full-chain microbenchmark

The table includes the original strided-gate materialization, norm, quant and
unchanged GEMM. Candidate consumes the strided gate directly. Each cell averages
A/B and B/A timing medians with8 warps,30 replays and3 distinct input buffers.
Reference and candidate retain all graph output buffers through timing.

| layer | tokens | reference us | fused us | change |
|---:|---:|---:|---:|---:|
| 0 | 4096 | 172.69 | 146.32 | -15.27% |
| 0 | 8192 | 390.07 | 285.28 | -26.86% |
| 0 | 16384 | 923.50 | 581.35 | -37.05% |
| 18 | 4096 | 173.09 | 146.82 | -15.18% |
| 18 | 8192 | 391.05 | 286.90 | -26.63% |
| 18 | 16384 | 925.94 | 584.51 | -36.87% |
| 38 | 4096 | 173.48 | 146.95 | -15.29% |
| 38 | 8192 | 392.09 | 286.56 | -26.92% |
| 38 | 16384 | 930.11 | 588.58 | -36.72% |

These measurements support an integration experiment. They do not establish that
every vLLM forward saves the same time: real scheduling, padding, compiler
specializations and launch/capture boundaries still require in-server validation.
The actual next MTP2 graph bypasses `_output_projection`; a hook there would not
activate. Integration must preserve the existing packed decode/state path and
bridge the fused FP8 result into the normal output-projection consumer.

## Failures retained

The original implicit-layout kernel fails one FP8 byte on a1025-token changed
input replay. Diagnostics isolate different RMSNorm reduction ordering before
BF16 rounding. The explicit implementation matches installed FLA's lane layout
and division/epsilon boundary. Original failure artifacts remain unchanged.

The first activation capture failed because the RPC serializer rejects a
TorchVersion subclass in worker metadata. It required targeted shutdown of the
owned engine and hung driver. The corrected capture uses a plain string and
bounded RPC calls; `activations_8k_v2` exited0 with clean owned-engine shutdown.
The failed directory is not a successful validation artifact.

## Reproduction and raw files

All commands run sequentially under the repository GPU lock in the pinned
qwen-prefill-dev container. No server was active alongside these GPU runs.

```bash
python3 verify_bench.py --run-gpu --mode verify --implementation explicit --tokens 3 16 31 1025 --warps 4 8 16
python3 verify_bench.py --run-gpu --mode both --implementation explicit --tokens 3 16 31 1024 8192 16384 --warps 4 8 16 --with-gemm --snapshot "$QWEN_SNAPSHOT"
python3 capture_activations.py --run-gpu --output-dir results/activations_8k_v2
python3 verify_bench.py --run-gpu --mode verify --implementation explicit --captured-dir results/activations_8k_v2 --warps 4 8 16 --with-gemm --snapshot "$QWEN_SNAPSHOT"
```

Set `QWEN_SNAPSHOT` to the local1c3f884bc99aac2524f6d49bcbac8c88401afd66 snapshot.
Capture output must be a fresh directory. JSONL and stderr records are
`explicit_initial_gpu.*`, `explicit_real_gpu.*`, `explicit_captured_gpu.*`;
capture manifests and hashes are in `activations_8k_v2/`. Tensor `.pt` files are
kept locally and excluded from git. Source/layout diagnostics are retained in
`explicit_layout_offline/`, `failure_01/`, `initial_sources/`, and
`reduction_replay_cpu.json`.
