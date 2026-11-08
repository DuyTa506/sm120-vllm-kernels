# Tiny-M rowwise swap prototype

This is an isolated extension of the existing same-byte row/channel epilogue,
not a claim that native UE8M0 quantization matches the Qwen checkpoint.

The source was copied only after the DeepGEMM owner confirmed it frozen. Before
patching, its full 6,945-entry fingerprint matched
`bbffbea38eb18e6abb84f7b860ae297e1b57b06d99729fbf6c8be0cbe5d112e2`.
The shared source still matches that fingerprint after CPU work.

The new `fp8_rowwise_gemm_nt_sm120_tiny` API accepts only M1..16. The original
M>=32 API is unchanged. Both orientations retain FP8 bytes and FP32 token/channel
scales, BF16 output, split-K=1 and no accumulation/alpha. The swapped call exchanges
operands and unit UE8M0 grids, then effective row/channel pointers. A distinct
`EpilogueRowChannelSwapped` marker performs `(acc * effective_column_scale) *
effective_row_scale`, preserving the original token-then-channel FP32 rounding
order. The existing transposed direct-store path performs the output mapping.

CPU validation completed:

- Extension build succeeded without exposing a GPU.
- NVCC compiled both the ordinary epilogue and swapped M1/3/8/16 instantiations,
  using exact measured native QKVZ policy template arguments and original JIT
  compiler flags. This checks device templates, not just host binding compilation.
- Four CPU contract fixtures passed, including a real non-associative FP32 scale
  counterexample, unchanged original API and exact tiny-QKVZ skinny dispatch.
- Captured layer0/9/29 payload SHA checks and skinny source/binary manifest checks
  pass; `_pick(M,12288,2048)` is `(1,4)` for all tested M, not `(2,4)`.

Future GPU smoke, only after the root grants a slot:

```bash
sg docker -c 'bash kernels/pdl-swapab/run_rowwise_swap_control.sh rowwise_swap_smoke_1 --layers 0 --ms 3 --iterations 12'
```

`replay_rowwise_swap.py` compares CUTLASS, the exact existing skinny candidate,
DG off and DG on. It verifies captured full-shape baseline reproducibility, then
checks sliced real activation rows against a fixed FP32 GEMM followed by token
then channel scaling. The gate is relative-L2 <=0.001 against CUTLASS and FP32
reference error no worse than 1.01 times baseline error (floor1e-6); these are
operator screens, not model-quality proof. CUDA graphs must match eager after a
second disjoint actual activation slice, changed token/channel scales and zero
scale. M0 and M17 must reject. Cold timing uses >=640MiB weights and reversed
backend orders. Unknown split-K remains a hard error, never a fallback.

The first granted layer0/M3 smoke now passes all numerical/replay/rejection
checks (results/rowwise_swap_smoke_1). DG swapped is3.61–3.87% faster than DG off,
but4.50% slower than the existing tiny-M skinny implementation. It is not promoted
to serving. No vLLM dispatch integration has been performed. Additional M/layers
remain untested; any further GPU work requires a new root slot.
