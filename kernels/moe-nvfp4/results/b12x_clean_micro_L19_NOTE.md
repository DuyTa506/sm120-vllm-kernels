The first line of `b12x_clean_micro_L19.txt` inherits a memory-accounting
error from the old verifier: 48 MiB of MMA-layout scale **views** per layer
were called copies and added to the scratch allocation. Installed FlashInfer
`convert_sf_to_mma_layout` uses only `view` and `permute`; the existing
swizzled scale storage is reused. `verify_bench.py` now asserts this storage
relationship and no longer double-counts it.

Measured wrapper allocation remains 579.652 MiB/layer, or 22.64 GiB for 40
unshared wrappers. Sharing saves 39 copies, about 22.08 GiB. The microbenchmark
timing rows and numerical comparisons are unaffected. The original output
is retained unchanged; `verify_cutlass_exact.py` subsequently confirmed storage aliasing on the
actual GPU-loaded weights of layers 0 and 19; see `cutlass_exact_gpu.jsonl`.
