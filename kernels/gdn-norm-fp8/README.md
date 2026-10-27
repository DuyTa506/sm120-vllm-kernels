# GDN norm + SiLU + token FP8 prototype

**Pending GPU verification.** These standalone files have no serving hooks.
They were prepared while GPU experiments were assigned elsewhere; no kernel
compilation, numerical comparison or timing has been run for this candidate.
CPU-only CLI, Python AST parsing and whitespace checks passed.
The source/trace analysis is in
[`../moe-nvfp4/GDN_FUSION_RESEARCH.md`](../moe-nvfp4/GDN_FUSION_RESEARCH.md).

`fused.py` implements one Triton CTA per token. Inputs are BF16 X/Z
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
unchanged CUTLASS output projection. Synthetic activations do not establish
model accuracy; captured activations and end-to-end serving checks remain
necessary even if all included cases pass.

Run these only after the GPU is available, inside the vLLM 0.32.0 environment
with this directory mounted:

```sh
# CPU-safe status and CLI checks:
python3 verify_bench.py
python3 verify_bench.py --help

# Numerical/graph validation, then timing, with synthetic weights:
python3 verify_bench.py --run-gpu --mode verify --warps 4 8 16
python3 verify_bench.py --run-gpu --mode bench --warps 4 8 16

# Real norm and output-projection weights from the mounted local snapshot:
python3 verify_bench.py --run-gpu --mode both --with-gemm --warps 8 \
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

Before integration, inspect generated Triton IR/PTX for the BF16 round-trip,
the per-head reduction layout, reciprocal and FP8 conversion, plus register
count and spills. Changing warp count can change reduction ordering. This
prototype provides a concrete experiment; exact numerical agreement and a
speed improvement are both unproven. A passing synthetic benchmark alone
does not authorize a production default or alter the current vLLM backend.
