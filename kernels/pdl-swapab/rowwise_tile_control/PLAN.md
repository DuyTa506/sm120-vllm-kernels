# Explicit 64×16 tiny-QKVZ alternative

The measured rowwise AUTO swap works numerically but is about4.5% slower than
existing skinny on real layer0/M3. This separate source snapshot tests a narrow
weight-row tile; it does not replace AUTO silently.

Supported forced policy only: effective M12288, N1..16, K2048, FP8 rowwise
operands, no accumulation, split-K1. It selects exactly one valid native
64×16×64 candidate and uses native storage/pipeline/launch derivation. Unknown
policy strings or unsupported shapes fail. The scale epilogue is byte-for-byte
unchanged from the measured rowwise prototype.

AUTO's128×16 policy has96 logical work tiles;64×16 has192. Both launch a
persistent grid of188 CTAs. The new trace records logical tiles, grid CTAs,
first-wave active work, shared memory, stages and finalized split-K separately.
Narrow storage is expected88320bytes/16stages; actual selected trace and generated
kernel parameters must confirm it on GPU.

CPU host-extension build and four NVCC tiny-M template instantiations pass.
Four CPU fixtures pass. Twenty runtime inputs are frozen in runner_freeze.json;
the original rowwise-swap freeze remains unchanged and valid.

The native split-K audit discovered that DG_PRINT_CONFIGS prints before final
split selection. Its old printed trace remains preserved. Native N2048×K4096
uses splitK4 in both orientations; native QKVZ uses splitK1. The new trace is
emitted after finalization, and offline analysis binds exact generated template,
metadata and cubin hashes. No timing/numerical data was changed by that audit.

Future root-granted smoke:

```bash
sg docker -c 'bash kernels/pdl-swapab/run_rowwise_tile_control.sh rowwise_tile_smoke_1 --layers 0 --ms 3 --iterations 12'
```

Five retained comparators: CUTLASS, existing skinny(1,4), DG off, DG AUTO swap,
and DG64×16 swap. Changed activation/token/channel scales and zero scale must
pass the same original operator gates. Unit UE8M0 scale creation explicitly uses
FP32. All signs of performance changes are reported; slower candidates remain
in the comparison. The first actual layer0/M3 GPU smoke now completes successfully numerically,
but the narrow candidate loses11.22–11.33% to AUTO and16.07–16.17% to existing
skinny. Policy traces and generated kernels confirm the intended implementation;
this is not a missing-activation result. See results/rowwise_tile_smoke_1. No vLLM
measurement or serving promotion exists for the narrow candidate.
