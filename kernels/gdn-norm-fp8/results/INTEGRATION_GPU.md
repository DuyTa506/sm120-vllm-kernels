# Stateful GDN integration smoke: pass, limited scope

`results/qwen_sequential/gdn_integration_2` exited 0 on `shared_exact_gdn`,
with MTP2, TP1, compiled Inductor and the GDN flag enabled. All30 expected
GDN layers became ready; cleanup verification confirmed 22 source files.
This establishes successful startup, capture and actual request execution for
this configuration. It does **not** establish paired quality/state equivalence
or an end-to-end speed improvement. FlashInfer control was disabled
(`FI_CONTROL=0`), and this was the diagnostic suite, not the seven-task matrix
or final agentic evaluation.

The saved resolved compilation configuration has20 splits: the completed
baseline 19 plus the new GDN op. It retains
`fuse_rope_kvcache_cat_mla=False`, `custom_ops=['none']`, mode3 and no Inductor
graph partition. Removing only the new split makes the recorded compilation
configuration equal to the completed `exact_tasks_1` baseline configuration.
This comparison concerns compilation settings, not equivalence of the entire
model/runtime or autotuner state.

## Compiled graph evidence

Cache `c35103b048` contains 30
`gdn_attention_core_norm_fp8_packed_experimental` calls and zero old packed GDN
core calls in the backbone FX. For every GDN layer, the following split graph
accepts FP8 Q `[T,4096]` and FP32 S `[T,1]`; Q passes through views directly to
`torch.ops._C.cutlass_scaled_mm` with S and the same layer's loaded output
weights/scales. There is no second quantization between the new producer and
that GEMM. The stateful CUDA implementation remains opaque to this FX check.

Graph SHA-256 values:

- Backbone `computation_graph.py`:
  `ccc694ec86c7de295957bf4ecab375261680bac030d515534f6c5169e692943b`
- Eagle head `computation_graph.py`:
  `73e2c0726196166cf2540bbe408d7a6292fc6cdbfef8caa757a6d617e4303fb2`

`integration_gpu_summary.json` records full cache paths, source hashes,
all 30 producer/consumer edges with line numbers and CUTLASS arguments,
resolved settings, capture evidence and compact diagnostic results.
`summarize_integration_2_cpu.py` reproduces those checks from the worktree root
without importing Torch/vLLM or accessing GPU/HTTP.

## Capture and request execution

The profile pass completed 63/63 piecewise graphs and both sampled FULL graphs,
including the previously missed derived MTP size. Final backbone capture
completed 63/63 piecewise and 26/26 FULL graphs. The speculator also completed
its 63 piecewise prefill,26 FULL prefill and 11 FULL decode captures. The log
reports completed graph capture and API startup; no unwarmed-GDN-shape failure
or reported GDN norm/quant-output inference JIT warning remains in this run.
Other inference JIT warnings occurred for logits/sampling and GDN prefill/core
kernels; their names and log lines are preserved in the JSON. This is another
reason not to treat the recorded diagnostic timings as a controlled speed test.

All four temperature-zero decode scenarios completed with zero failed requests
and exact saved input/output lengths:

| Concurrency | Measured requests | Input tokens each | Output tokens each |
| ---: | ---: | ---: | ---: |
| 1 | 8 | 128 | 512 |
| 4 | 24 | 128 | 512 |
| 16 | 96 | 128 | 512 |
| 64 | 384 | 128 | 512 |

That is 512 measured decode requests, excluding warmups. Successful completion
and length checks do not compare generated token identities against baseline.

The NLL diagnostic completed 10 requests in one round, each with its own cache
salt. Every scored logprob, NLL and last-half NLL is finite; each request
returned one completion token and scored exactly prompt_tokens−1 positions.
Prompt lengths span 669–15225 tokens; eight exceed the 1024-token fusion
threshold and two are smaller. Thus both small-input and longer-prefill
requests executed, but no per-request kernel trace or explicit mixed-state
parity assertion was collected. One round supplies no within-configuration
repeat-spread estimate. No historical NLL comparison is used to claim quality
equivalence here.

## Failures retained and remaining checks

The first integration run reached the compiled QA path but failed during FULL
capture because the original 63-size warmup list omitted 18 rounded MTP sizes.
The corrected source-derived planner warms 81 sizes and retains the diagnostic
membership guard. Its configuration generator was also corrected to preserve
the two default KV-update split ops and disabled MLA rope fusion. The failed
run/source snapshot remains in `gdn_integration_1`; details and red/green CPU
regressions are in `CAPTURE_WARMUP_FIX.md` and `SPLIT_CONFIG_FIX.md`.

The earlier activation-capture RPC failure is separate: its TorchVersion
serialization bug and owned-process cleanup were preserved, then capture v2
and exact standalone captured-input checks passed as documented in CAPTURE.md.

Remaining adoption evidence includes matched baseline quality and serving
comparisons, decode regression measurement, and targeted mixed-batch accepted
state/checkpoint parity plus integrated padded-Q/S exactness where required.
The successful smoke does not promote the default-off experiment to a default.
