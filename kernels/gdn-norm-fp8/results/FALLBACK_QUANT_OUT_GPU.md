# Compiled-output FP8 fallback evidence

The isolated wrapper passes 60 numerical cases and 36 changed-input graph
replays over T=1/2/3/12/31/64/128/192/256/1024/8192/16384. FP8 bytes, FP32
scale bits and the unchanged real layer18 output GEMM match exactly.

Both generated specializations (size1 and dynamic sizes >=2) contain one
fused Triton launch writing directly to caller-owned Q/S. Their call bodies
allocate no intermediate Q/S and launch no separate destination copies.
The normal generated input-alignment guard remains; tested inputs are aligned.
Raw code is in `fallback_quant_out_gpu.stderr`; extracted call bodies, hashes,
all timings and counts are in `fallback_quant_out_gpu_summary.json`.

Small-shape CUDA-graph timings below cover the complete quantization-to-output
chain, including the baseline's two copies. Values average the two ABBA-position
medians (30 replays, 3 distinct buffer pairs; reference temporaries retained).
They exclude Python overhead, GDN core/norm and GEMM. **Native quantization
without the two separate copies was not timed as a comparator.** These figures
do not establish a decode speedup versus the original serving path.

| Tokens | Native quant + separate Q/S copies (us) | Compiled caller-output quant (us) |
| ---: | ---: | ---: |
| 1 | 4.245 | 2.949 |
| 2 | 4.203 | 3.008 |
| 3 | 4.237 | 2.920 |
| 12 | 4.243 | 2.928 |
| 31 | 4.581 | 2.920 |
| 64 | 4.365 | 3.056 |
| 128 | 4.749 | 2.923 |
| 192 | 4.915 | 3.517 |
| 256 | 4.907 | 3.568 |

The wrapper calls the original raw `QuantFP8.forward_native(x, None, None)`,
then copies into provided outputs in the same compiled graph. Both the verifier
and overlay use `torch.compile(..., dynamic=True, fullgraph=True)` with default
Inductor backend/options. The original quantizer object remains unchanged in
the overlay for the oracle and ordinary Tensor input path.

The isolated overlay now uses this wrapper in its post-load startup lifecycle:
compile and warm T1/T2/maxT/all graph buckets, then enable the producer. The
wrapper module is a byte-for-byte copy of the GPU-tested source; its original
docstring predates validation. Stateful GDN graph startup, prefill/decode
quality and serving performance remain separate integration checks.
