# Initial integration capture failure and CPU fix

The first owned integration run, `results/qwen_sequential/gdn_integration_1`,
failed before serving requests. All 30 GDN layers reported ready, and root's
compiled-FX review found 30 new packed ops feeding QA into CUTLASS without
requantization. PIECEWISE graph profiling completed; FULL profiling completed
one of its two sampled descriptors and then raised the retained unwarmed-shape
guard. This is a failed integration run, not serving validation.

The exact installed MRV2 source explains the missing sizes:

- `v1/worker/gpu/model_runner.py` derives `decode_query_len` from two draft
  steps plus the model state's one newly sampled token: query length 3.
- `CudaGraphManager._init_candidates` in
  `v1/worker/gpu/cudagraph_utils.py` creates uniform FULL descriptors with
  `round_up(configured_tokens, 3)`. It bounds them by 64 requests (192 tokens)
  and the configured maximum capture size. PIECEWISE uses original sizes.
- MRV2 does **not** rewrite `compilation_config.cudagraph_capture_sizes`.
  The overlay had warmed only this original list (63 sizes including T1/T2/maxT).
- Profile-only capture samples the two largest FULL descriptors, 192 then
  186. The first was warmed; 184→186 was not. The old error omitted actual T,
  so identifying 186 is a source-derived inference consistent with the trace,
  rather than a token value recorded in that error.

`audit_capture_sizes.py` executes the installed enum, descriptor dataclass and
candidate-builder method extracted with AST. It imports neither Torch nor
vLLM. The saved source-hashed result is `capture_sizes_cpu_audit.json`.
It identifies 18 missing FULL sizes:
`3,6,9,18,33,42,57,66,81,90,105,114,129,138,153,162,177,186`.

The fix warms the union of original sizes and bounded rounded FULL sizes
(81 sizes for this configuration) during the existing post-load lifecycle.
The runtime membership guard remains; its error now includes layer, padded T,
actual-token count, prefill count and warmed sizes. Eligibility requires fixed
MTP without DBO; unsupported dynamic speculative schedules retain the original
producer path. No graph configuration, runner lifecycle or attention-state
implementation is changed.

The regression test failed before the fix (`capture_warmup_regression_red.log`)
and passes afterward (`capture_warmup_regression_green.log`). It compares the
warmup set against the installed MRV2 builder, checks capture/request bounds,
and retains a negative test for an unwarmed capture shape. The full relevant
CPU suite is recorded in `integration_cpu_preflight.json`, with CUDA hidden,
uninitialized and runtime entrypoints forbidden. GPU recapture and real
prefill/decode correctness remain pending; the CPU result is not a claim that
integration now succeeds.
