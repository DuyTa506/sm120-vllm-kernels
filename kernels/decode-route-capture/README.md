# Actual V2 decode routing capture

Status: **target capture ran; exact output equivalence failed and remains
unresolved**. The c1 control and target-only runs generated 192 tokens each but
differed at token zero. Their independent autotuning selected different tactics
in 51/132 entries. The signed target artifact passes structural checks; full
target/draft capture is pending. This checkpoint supports no optimization
adoption. See [the recorded result](RESULTS.md) for scope and proposed controls.

The later fixed-reference plain control `fixed_plain_c1` passed: 192 output
tokens, all 132 tactics unchanged at every boundary, complete observed lookup
coverage and clean shutdown. Native-aux/full capture and output comparisons
under this reference remain pending.

The 57 CPU analysis/launcher/driver/import/lifecycle tests pass.
Hooks were inspected against installed vLLM 0.32.0 / Torch 2.13.0+cu130 /
FlashInfer 0.7.0.post1 sources in `qwen-prefill-dev`. The target path completed
compilation, graph replay, capture, export and shutdown. Draft graph compatibility
and output preservation remain unvalidated.

This is instrumentation for Qwen3.6-35B-A3B NVFP4 Fast on one SM120 GPU with
CONFIGnext, CUTLASS NVFP4, fixed MTP-2, and TP/PP/DP/PCP/DCP=1. It does not
replace expert choices, recompute the router, quantize the draft, or compact
model inputs. It uses existing production-recorded agentic prompt token IDs;
generated text is never interpreted as a tool call. No HTTP server is needed.
All changes are local to this directory and temporary in-process Python hooks.

## Reproduction, in the root agent's sequential GPU queue

From the repository root, first validate the target path, then add draft capture.
The launcher acquires the repository GPU lock and starts its own temporary
container. It does not stop or edit another container. Keep the same options
and prompt offset for all three runs. The first two commands perform the minimal
control/target capture/output comparison; the third adds both draft phases:

```bash
sg docker -c 'bash kernels/decode-route-capture/run.sh control_c1 --control --concurrency 1'
sg docker -c 'bash kernels/decode-route-capture/run.sh target_c1 --target-only --concurrency 1 --compare-output /work/kernels/decode-route-capture/results/control_c1/outputs.json'
sg docker -c 'bash kernels/decode-route-capture/run.sh full_c1 --concurrency 1 --compare-output /work/kernels/decode-route-capture/results/control_c1/outputs.json'
```

These are separate queued jobs. A failed control/target comparison requires
inspection of the controls before further diagnostic capture. The completed checkpoint used `review_control_c1`
and `review_target_c1`; it stopped before the full run and released the GPU.

After c=1 works, repeat c=4/16/64 with fresh tags and matching controls; repeat
with `--prompt-offset 32` for different recorded contexts. Defaults are 192
generated tokens, at most 64 pure-decode steps captured, 512 padded rows, and
GPU utilization 0.60. The same model flags, graph sizes and four CONFIGnext
module mounts as `scripts/serve.sh` are used. `GPU`, `GPU_LOCK`, `HF_CACHE`, and
`IMAGE` can be overridden. No server lifecycle or serving harness is changed.
Each tag has isolated compile caches; compilation may therefore take time.
The local `.gitignore` excludes `results/*/cache/` and local capture artifacts.
Existing result directories are rejected so a retry requires a fresh tag.

When a parent wrapper already holds the **same** `GPU_LOCK`, pass
`GPU_LOCK_HELD=1` inside the `sg docker -c` command. That mode checks that the lock
is occupied without reacquiring it; the parent must retain ownership until the
launcher exits. Direct sequential invocations above need no borrowed-lock flag.
The held-lock check cannot identify its owner, so the flag is an explicit caller
assertion. CPU tests verify that it neither deadlocks nor releases the parent lock.

Before starting a container, the launcher refuses any existing compute PID on
the selected GPU. It records the invocation, exit code, GPU telemetry before and
after, and compute-process snapshots, including failed runs. Cleanup only removes
the container ID returned through this invocation's `--cidfile`. The driver
exports raw routes before calling its own v0.32
`llm.llm_engine.engine_core.shutdown()` handle; shutdown results are recorded in
`shutdown.json`, and a cleanup error cannot hide the original inference error.
Constructor failure leaves no complete engine handle, so the disposable
container is the remaining cleanup boundary.

The default trajectory file is
`results/agentic/trajectories_short_production.json.xz`. Prompt selection
interleaves tasks by recorded turn index. It selects distinct recorded turns,
rather than repeating one prompt to fill concurrency. `--trajectory` can select
the existing long-session recording. Prompt task/turn, length and SHA256 are
saved. A context exceeding 131072 tokens after generation is rejected.

Do not treat the control or instrumented run as a latency/throughput benchmark.
Capture adds device copies, synchronization at draft phase boundaries, memory
pressure, and native auxiliary-output handling. It can change asynchronous
batching and available KV capacity. `--compare-output` fails if prompts or output
token IDs differ from the chosen control; it does not prove identical batching.

The subsequently prepared [control and tactic-import options](CONTROLS.md) add
`--native-aux-control`, optional `--autotune-json` with observed-choice
verification, and separate baseline token-distance reporting. They are CPU-tested
and GPU-unrun. Default inference options are unchanged. Completed inference now
gets structural route analysis even after an exact-token mismatch; overall
failure is preserved, with diagnostic analysis saved separately.

The separate [serving cache-control plugin](SERVER_INTEGRATION.md) preloads a
frozen reference table before model loading, replays native cache lookup without
retuning during warmup, and verifies effective snapshots after graph capture and
at shutdown. The parent serving runner enables it with `FI_CONTROL=1`.
It installs no route hooks or per-lookup serving observer; Inductor settings
remain independently selected and audited.
The first native-tuning import failed before measured requests because cold-L2
dense FP4 tuning ignored legacy v1 entries. Revised serving replay and the
30-second engine drain passed startup/shutdown gates on the lifecycle smoke and
the full exact/scalar pair. The separate offline lifecycle wrapper now has CPU
coverage and passed the plain-control GPU run. Native-aux/full-draft runs remain pending.

## What is captured

`raw_routes.pt` is the authoritative artifact: each phase's route tensor remains
`torch.int32`, including `-1` IDs and all graph-tail/rejected rows. No route is
converted to uint8. `records.jsonl.gz` is an exact integer-list view for a
standard-library-only CPU analyzer. The raw artifact is written before analysis,
so a failed invariant does not erase evidence.

| Phase | Inputs and rows | Capture boundary |
|---|---|---|
| `target_verify` | Every executed verifier row, including rejected speculative candidates, plus graph tail | Native raw capturer buffer in `AuxOutputWorkerConnector.prepare_output` |
| `draft_prefill` | First MTP pass, with each request's retained/rejected suffix and graph tail | `MTPSpeculator.on_prefill_end` after graph replay |
| `draft_decode` | Second MTP pass, one logical row per request plus graph tail | `MTPSpeculator.on_multi_step_decode_end` after graph replay |

Each record has a target-step ID, phase, request IDs, original layer IDs, raw
expert IDs, actual input token IDs/positions, logical query boundaries, real and
padded row counts, and rejection/sample counts. Target records retain the
target's GPU padding mask and both CPU/GPU query boundaries. Draft records
separately retain the **actual router mask presence and contents** and the
**logical graph-tail classification**; an absent router mask is never fabricated
from logical padding. First-draft rejection boundaries come from the GPU
`num_rejected` passed to that proposal. Second-draft rows have zero local
rejection count, and also retain the target's rejection count as separate data.

MTP-2 has one remaining draft decode forward, so the phase-end snapshot captures
that whole second phase. This probe rejects other speculative lengths,
adaptive/dynamic verification, expert load balancing, context parallelism and
microbatch overlap. It captures pure decode, not prefill/decode mixtures. Missing
draft phases cause a failed full-capture run; `--target-only` explicitly writes
`draft_capture=disabled_target_only` and does not invent draft data.

## Why the hooks preserve routing and graph pointers

The current V2 target runner passes `input_batch.is_padding` to its forward
context; graph capture also records the persistent mask. The native
`BaseRouter._select_experts` capture callback sees actual top-k IDs. Native aux
delivery later narrows these IDs and removes rejected suffixes, so this probe
reads `connector._capturer.device_buffer` directly instead of the public returned
routes. It clones snapshots without replacing the native buffer and verifies
that its device pointer is unchanged.

The MTP loader binds a separate fixed buffer to the existing draft router's
capture callback before compilation. The callback only copies IDs and records
mask metadata. `_run_model` records copies of input IDs, positions, row count
and draft step into persistent buffers **before** model execution. These device
operations are part of graph capture/replay. This is necessary because
`_prefill` overwrites positions and `_generate_draft` updates inputs after model
execution. A Python-only hook after replay would otherwise see mutated inputs.

At each outer phase boundary, the probe checks stable buffer pointers, matching
per-layer row counts, and the expected draft-step scalar before cloning. These
checks synchronize outside the captured graph. Only a real `propose` call
associated with a captured target step is recorded; profiling and graph-capture
dummy runs are ignored. No source tensors are written by the probe.

Inspected package paths (relative to the container's
`/usr/local/lib/python3.12/dist-packages/vllm/`):

- `v1/worker/gpu/model_runner.py:2181,2239`: target snapshot and same input-batch
  object passed to `speculator.propose`.
- `distributed/aux_output_connector/worker.py:138,162`: raw target snapshot
  opportunity and completed acceptance metadata. Public `_commit_output` strips
  rejected suffixes at line 228; that filtered output is not used here.
- `model_executor/layers/fused_moe/router/base_router.py:293`: top-k capture
  before EPLB mapping; EPLB is disabled for this probe.
- `v1/worker/gpu/spec_decode/mtp/speculator.py:15,44,57`: draft loader and phase
  hooks. `target_dependent_ar/speculator.py:294,362` calls these after replay.
- `target_dependent_ar/speculator.py:367`: actual draft model input boundary;
  lines 376-390 only supply a MoE mask when PCP is present. The single-GPU draft
  mask gap is an observed source property, not a measured cost yet.
- The same file, lines 702-730: first-draft preparation subtracts rejected rows
  per request, shifts accepted-prefix IDs, and leaves unused suffix storage.
  Lines 477 and 657-670 mutate inputs after execution, which is why the extra
  graph-recorded input copy is needed.
- `v1/worker/gpu/input_batch.py:507-553`: pure-decode sampler counts satisfy
  `num_sampled + num_rejected = num_logits = scheduled_draft_tokens + 1`.
- `v1/engine/core_client.py:401-403`: `InprocClient.shutdown()` detaches its
  finalizer and shuts down only its own engine. `v1/engine/core.py:824-839`
  destroys distributed resources, which is why raw export happens first.

`manifest.json` saves package versions, source hashes, runner identity, the
configuration, selected prompts, and live expert geometry/weight tensor sizes.
The launcher also saves image ID and CONFIGnext overlay/probe source hashes.

## CPU analysis and self-check

```bash
python3 -m unittest discover -s kernels/decode-route-capture -p 'test_*.py' -v
python3 kernels/decode-route-capture/analyze.py kernels/decode-route-capture/results/full_c1
```

The analyzer verifies signed IDs, distinct top-k IDs, shapes, target masking,
actual draft mask/route agreement, request-local rejected suffixes, pure-decode
acceptance counts, and complete/unique step/phase pairs. An all-zero unwritten
native buffer cannot pass as valid top-k selection. For each layer and logical request group
it reports expert unions, adjacent/lag-2 intersections and overlap with that
request's previous captured group (with the step gap). It also reports
draft-prefill/decode union/intersection for the **same draft layer**, separately
from target weights. A numerical expert ID in two layers is not shared storage.

Geometry comes from each live MoE module and its loaded weight/block-scale
tensors. For H=2048, I=512, E=256, top-k=8, target NVFP4 payload per expert is
`3*H*I*(1/2+1/16) = 1,769,472 bytes`. A BF16 draft expert has
`3*H*I*2 = 6,291,456 bytes`. Stored tensor sizes are recorded and used for byte
accounting, so layout/alignment is inspectable. Small global scales,
activations, shared experts, attention, and dense layers are additional traffic.

Target byte accounting includes **all verified candidates**, even those later
rejected. Accepted-prefix-only unions are also shown for diagnostics but are
not proposed as legal target work elimination. Draft-only potentially avoidable
experts are `(rejected ∪ graph_tail) - required`; this is an upper-bound lead,
not proof those rows can be removed without changing attention/KV semantics.

`sampled_output_tokens` is the actual target sampler count, and acceptance uses
recorded proposed/rejected counts. At request termination, sampler counts can
differ from client-emitted tokens; the complete generated output IDs are saved
in `outputs.json`. The analyzer labels its denominator accordingly. Unique
weight payload is **not measured DRAM traffic**: repeated token tiles and cache
residency are not inferred from route overlap. No latency gain is reported.
