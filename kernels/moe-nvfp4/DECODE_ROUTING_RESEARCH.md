# Decode routing: current V2 runtime and capture plan

Inspected 2026-10-27, read-only, in `qwen-prefill-dev`. No inference, GPU
benchmark, backend change, or harness change was performed for this note.
Installed vLLM reports `0.32.0`, commit `g5df63a720`; FlashInfer is
`0.7.0.post1` in the current startup log. All package paths below are relative
to `/usr/local/lib/python3.12/dist-packages/` **inside that container**.

## Findings and priority

1. **Target graph padding does not justify an extra-expert byte model.** The
   current V2 runner provides a persistent padding mask to the router. CUTLASS
   filters invalid expert IDs before computing per-expert problem sizes.
   A padded target batch of four rows can still have only three routed rows.
2. **Actual MTP temporal routing is unmeasured.** Three adjacent verification
   positions can share experts, and the existing random subsets of prefill
   hidden states do not measure that sharing. Count the union of experts
   across all executed verification rows, including subsequently rejected
   positions, separately for each layer and step.
3. **Draft padding is a separate, concrete lead.** Current V2 MTP omits the MoE
   padding mask on the ordinary single-GPU path. Its first draft forward also
   includes rejected positions. This affects the BF16 draft MoE, not all forty
   NVFP4 target layers. Its savings are unmeasured and should not displace the
   active prefill work without a route/cost capture.
4. Preserve CUTLASS NVFP4. The plausible target optimization is reducing routing,
   expansion, and descriptor setup overhead; a DeepGEMM MXFP4 substitution does
   not test the same quantization format or end-to-end work.

## Authoritative runtime path

The current `results/shared_b1/startup.log:31` explicitly says
`Using V2 Model Runner`; the loading/capture records at lines 32, 69, and 99
come from `model_runner.py`. The separate phrase `V1 LLM engine` is the engine
generation, not the GPU model-runner version. This run has TP=PP=DP=PCP=1,
MTP-2, and the BF16 draft fallback (`unquantized.py` log at line 57).
It is a b12x-hybrid prefill experiment: small target batches use CUTLASS.

### Target: graph tail is masked

- `vllm/envs.py:1661`: `VLLM_MOE_SKIP_PADDING` defaults to `"1"`.
- `vllm/v1/worker/gpu/input_batch.py:19-35,87-92`: input buffers and each input
  batch carry an `is_padding` tensor.
- `vllm/v1/worker/gpu/model_runner.py:1364-1367`: capture setup marks the real
  prefix false and graph tail true. Its capture forward context supplies the
  mask at line 1539. Real eager/piecewise forwards supply
  `input_batch.is_padding` at line 2026; full graph replay occurs at line 2006
  using the captured buffers. Thus inspecting only the eager call is inadequate.
- `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:19-39`:
  `_get_padding_mask` slices `get_forward_context().is_padding`, and
  `vllm_topk_softmax` passes it to the compiled `ops.topk_softmax`.
  The runtime contract being checked is padded IDs `-1`; the compiled CUDA
  implementation is not shipped as source in this wheel, so the capture below
  explicitly asserts this instead of treating a source inspection as a GPU test.
- `.../experts/flashinfer_cutlass_moe.py:365-369`: sends those `topk_ids` and
  `topk_weights` directly to the FlashInfer CUTLASS call.

Within `flashinfer/data/csrc/fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh`:

- Lines 412-452: the fused expert-map builder validates
  `expert >= start_expert && expert < end_expert`; `-1` goes into the invalid
  bucket. Lines 4265-4285 select fused sorting or the three-step fallback.
- Lines 1403-1417: `gemm_m = offsets[e+1] - offsets[e]`, then both grouped
  problem shapes use this exact count. Inactive experts get zero rows.
- Lines 1615-1624: expansion loops over
  `expert_first_token_offset[num_experts_per_node]`, the valid routed-row count.

This removes padded expert weight traffic, not every cost of padding. Router
GEMM, input quantization, metadata traversal, and launch/grid overhead can still
depend on the padded tensor shape. Also distinguish graph padding from the
unused lanes of a 32-token MMA tile: the latter remain for sparse real experts.

### MTP: do not confuse verifier rows, rejected rows, and graph tail

- `vllm/v1/worker/gpu/spec_decode/__init__.py:63-66` chooses `MTPSpeculator`.
- `.../mtp/speculator.py:12` inherits `TargetDependentARSpeculator`.
- `.../target_dependent_ar/speculator.py:204-209`: draft position 0 retains the
  target query length, including rejected positions, to avoid a CPU/GPU sync.
  Lines 225-236 prepare inputs and last-token indices; lines 276-294 run the
  first draft forward; lines 306-362 prepare and run the remaining draft step.
- Crucially, `_run_model`, lines 376-390, initializes `is_padding = None` and
  supplies a mask **only when `self.pcp_manager is not None`**. The current
  single-GPU PCP=1 path therefore does not get the target's ordinary mask.
- Target verification must compute speculative candidates before acceptance is
  known. Their rejected suffix is real work for byte accounting. The next
  draft pass's retained rejected suffix is a different potential optimization.
- `mtp/speculator.py:27-35` has optional `index_share_for_mtp_iteration` hooks
  requiring model-specific `set_skip_topk`/`compact_topk_indices`. Their name is
  not evidence of Qwen MoE expert reuse; the Qwen MTP class inspected here does
  not implement these methods. Do not force expert reuse across tokens.

The legacy `v1/worker/gpu_model_runner.py:4187-4199` and
`v1/spec_decode/llm_base_proposer.py:590,746` show a similar target/draft
distinction, but they are not the authoritative execution path for this run.

## Scheduling and the appropriate expert-byte model

`flashinfer/.../moe_gemm/launchers/moe_gemm_tma_ws_launcher.inl:677-685` uses
grouped GEMM and `RasterOrderOptions::AlongN`. In the bundled CUTLASS,
`include/cutlass/gemm/kernel/tile_scheduler.hpp:401-416` maps SM120 grouped
scheduling to `PersistentTileSchedulerSm90Group`.
`sm90_tile_scheduler_group.hpp:328-391` warp-scans groups and prefix sums their
tile counts; `advance_to_next_work` at line 462 increments by the persistent
grid size. Empty groups have no GEMM tiles, but the scheduler can still traverse
their metadata. This is scheduling across tiles, not one CTA pinned to each
expert. Changing expert order does not reduce the set of required weights.

For this checkpoint, `moe_common.py:10` gives E=256, top-k=8, H=2048, I=512.
Gate/up/down have `3*H*I = 3,145,728` weight elements per routed expert:

```
b_expert = 3*H*I * (1/2 + 1/16)
         = 1,769,472 bytes = 1.6875 MiB
```

This includes packed FP4 weights and one FP8 block scale per 16 elements;
tiny global scales, alignment, descriptors, activations, shared experts,
attention, dense weights, and the draft model are additional costs.

Let `S[l,r,j]` be the actual eight experts for request r, verification position
j at layer l. For a target step with its actual scheduled row count:

```
U[l] = size(union of S[l,r,j] over every nonpadding executed row)
target unique bytes = b_expert * sum_l U[l]
conditional cold-weight time = target unique bytes / achieved DRAM bandwidth
bytes per emitted token = target unique bytes / actual emitted tokens this step
```

Use all verified positions, including rejected candidates. MTP-2 usually has
three verifier rows/request, but startup, prefill mixtures, scheduler changes,
and partial proposals require the recorded query lengths. Never infer real
rows from the CUDA-graph bucket alone. Divide by observed emitted-token count,
not by three or a single aggregate acceptance percentage.

For c=1, identical experts across three positions give U=8; disjoint sets give
U=24. The forty-layer cold-weight estimates at an **assumed** 1.6 TB/s are
0.354-1.062 ms. They are not a measured latency range, and they assume the same
U at every layer solely for illustration. Uniform independent top-8 sets give
`256*(1-(1-8/256)^(3*c))`:

| Concurrency | Uniform-independent U/layer | Forty-layer time at assumed 1.6 TB/s |
|---|---:|---:|
| 1 | 23.258 | 1.029 ms |
| 4 | 81.104 | 3.588 ms |
| 16 | 200.230 | 8.858 ms |
| 64 | 255.423 | 11.299 ms |

**These are null-model examples, not forecasts.** Consecutive tokens may share
experts; different requests also may share experts. Measure the union directly,
plus intersections at lags 1/2 and across target steps. Cross-layer expert IDs
refer to different weight tensors and cannot be deduplicated. Across-step route
overlap does not guarantee L2 hits: forty target layers, dense weights and draft
work intervene. For small decode, read each active expert once is a useful
conditional cold-weight lower bound. For prefill, multiple token tiles may
reread weights, so unique bytes alone severely underdescribe actual traffic.

One BF16 draft expert with the same dimensions holds 6,291,456 bytes (6 MiB).
Draft pass 0 and pass 1 use the same draft-layer weights, so their measured
temporal union can matter to cache reuse. Keep this byte model separate from
the target's 1.6875 MiB/expert model; identical numerical expert IDs in target
and draft layers are different weights.

Existing evidence limits: `deepgemm-sm120/bench_ours_cold.py:27-44` selects random
prefill hidden rows and rotates six weight copies. It is useful for synthetic
cold-vs-warm behavior, not temporal MTP routing. `pdl-swapab/README.md:203-204`
admits M=4 uses a real fourth row but then assumes vLLM also routes padding;
the latter assumption is obsolete for the target path above. No saving from
removing that fourth real token should be claimed for an already-masked target.

## Runnable raw-target capture, deferred until the GPU is free

The installed `BaseRouter._select_experts` calls `capture_fn(topk_ids)` after
actual top-k and before EPLB mapping (`router/base_router.py:293-305`). At TP1,
EP1 with no EPLB this is exactly the logical routing sent to CUTLASS.
`RoutedExpertsCapturer.capture` copies IDs into a preallocated int32 device
buffer, so it is compatible with graph replay. A Python-only forward hook
installed after graph capture will not reliably observe replayed routing.

Use native `aux_output_config={"enable_return_routed_experts": True}` to bind
capture before compilation. **Do not use only the public returned routes**:
`snapshot_routing_data` narrows to uint8 for E=256 (so -1 becomes 255),
`AuxOutputWorkerConnector.prepare_output` excludes graph-tail rows, and
`_commit_output` line 228 removes rejected suffixes. Those outputs answer a
different question.

The following standalone diagnostic preserves the original IDs, records all
executed target verification routes and graph padding, and joins acceptance
counts. Put the fenced Python into `/tmp/decode_route_probe.py`, then run it in
the prepared CONFIG=next container after the sequential GPU queue allows it:

```
python3 /tmp/decode_route_probe.py 1 /tmp/routes_c1.pt
```

Repeat with 4,16,64 and varied prompt families. The code is a source-derived
capture design; it has been syntax-checked but **not run on the GPU**. It uses
an in-process engine so the diagnostic monkeypatch stays in the worker.
CONFIGnext's patched modules and environment must already be present; the
script does not install those patches. Preserve the same baseline/backend
configuration for A/B, including calibrated CUTLASS quantization. The explicit
CUTLASS backend here isolates the routing study from the b12x prefill experiment.

```python
import os
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
import sys
import torch
from vllm.distributed.aux_output_connector.worker import AuxOutputWorkerConnector

records = []
limit = 256
original_prepare = AuxOutputWorkerConnector.prepare_output
original_process = AuxOutputWorkerConnector.process_output

def prepare(self, batch):
    pending = original_prepare(self, batch)
    if (pending is not None and not batch.has_prefill
            and len(records) < limit):
        n = batch.num_tokens_after_padding
        # Clone on the model stream before the next target step overwrites it.
        record = dict(
            step=len(records), request_ids=list(batch.req_ids),
            query_start_loc=batch.query_start_loc_np.copy(),
            num_tokens=batch.num_tokens,
            num_tokens_after_padding=n,
            num_computed_tokens=batch.num_computed_tokens_np.copy(),
            ids=self._capturer.device_buffer[:n].clone(),
            is_padding=batch.is_padding[:n].clone(),
            input_ids=batch.input_ids[:n].clone(),
            positions=batch.positions[:n].clone(),
        )
        records.append(record)
        pending._routing_research_record = record
    return pending

def process(self, pending):
    record = getattr(pending, "_routing_research_record", None)
    if record is not None:
        record["num_sampled"] = pending.num_sampled.copy()
        record["num_rejected"] = pending.num_rejected.copy()
    return original_process(self, pending)

AuxOutputWorkerConnector.prepare_output = prepare
AuxOutputWorkerConnector.process_output = process

from vllm import LLM, SamplingParams
c = int(sys.argv[1])
out = sys.argv[2]
sizes = [int(x) for x in open("/work/scripts/cudagraph_sizes.txt").read().split()]
llm = LLM(
    model="unsloth/Qwen3.6-35B-A3B-NVFP4-Fast",
    language_model_only=True, max_model_len=131072, max_num_seqs=64,
    max_num_batched_tokens=16384, gpu_memory_utilization=0.78,
    enable_chunked_prefill=True, enable_prefix_caching=True,
    async_scheduling=True, block_size=2176, seed=0,
    speculative_config={"method": "mtp", "num_speculative_tokens": 2},
    compilation_config={"cudagraph_capture_sizes": sizes},
    kernel_config={"moe_backend": "flashinfer_cutlass"},
    aux_output_config={"enable_return_routed_experts": True},
)
prompts = [
    f"Request {i}. Explain a distinct practical example of locality in computer "
    "systems, then reason through its performance tradeoffs in detail."
    for i in range(c)
]
outputs = llm.generate(prompts, SamplingParams(
    temperature=0, max_tokens=256, ignore_eos=True))
torch.cuda.synchronize()
assert records, "No decode capture: check V2 runner and aux-output binding"
for record in records:
    for name, value in list(record.items()):
        if isinstance(value, torch.Tensor):
            record[name] = value.cpu()
    assert "num_rejected" in record, "Acceptance metadata was not joined"
    padded = record["is_padding"].bool()
    assert (record["ids"][padded] == -1).all(), "Unexpected padded target routing"
    real = record["ids"][~padded]
    assert ((real >= 0) & (real < 256)).all()
    record["unique_experts_per_layer"] = [
        int(torch.unique(real[:, layer]).numel())
        for layer in range(real.shape[1])
    ]
torch.save(dict(records=records,
    output_token_ids=[list(o.outputs[0].token_ids) for o in outputs]), out)
print(out, "steps", len(records))
```

This deliberately stores small decode snapshots on GPU until the run finishes;
it never changes top-k, weights, router probabilities or quantization scales.
Capture adds memory traffic and can change asynchronous batching/timing. Use it
for route statistics, check generated token IDs against an uninstrumented
deterministic run, and measure latency separately without capture. Include
boundary batches and naturally terminating requests in a second study; do not
mistake the fixed-output-length diagnostic for a serving workload.

### Extend capture to the two draft phases

Before draft compilation, wrap `MTPSpeculator.load_draft_model` to attach a
separate per-layer int32 route buffer via each `MoERunner.router.set_capture_fn`.
Use the native capturer's device-copy pattern; store each layer's row count in
a persistent GPU scalar too, so no Python replay hook is needed. Also copy the
forward-context mask (or record mask-absent) into the buffer. Do not reuse the
target capture buffer or silently remap an out-of-range draft layer index.

Wrap `TargetDependentARSpeculator.propose` only to identify real calls and record
its `input_batch`, `num_sampled`, and `num_rejected` metadata. Snapshot the draft
buffer in `MTPSpeculator.on_prefill_end` and `on_multi_step_decode_end`, after
calling the original hooks. These boundaries execute after full graph replay
too (`target_dependent_ar/speculator.py:294,362`). For MTP-2 there is exactly one
remaining decode forward, so the second snapshot is sufficient; larger MTP
requires per-iteration buffers. Ignore graph-capture/dummy lifecycle calls.

Record `last_token_indices`, draft input positions/IDs, `current_draft_step`,
actual and padded row counts, query starts, per-request rejected counts, and
the active runner/backend/dtypes. Classify ordinary graph tail separately from
each request's rejected suffix; their positions need not form one global
prefix. Estimate potentially avoidable draft weight bytes as
`6 MiB * |experts(padded-or-rejected) minus experts(required)|`, per draft layer
and phase, then measure cache reuse between phases. An eight-expert padded row
does not necessarily introduce eight new experts.

## Two scoped candidates after capture

1. **Prioritize NVFP4 routing/setup fusion or reuse of immutable descriptors.**
   Keep actual top-k, exact per-expert counts, CUTLASS weights/scales, and both
   GEMMs. The fused sort already exists; avoid reimplementing it blindly.
   Inspect whether expansion plus pointer/stride/problem-shape setup can share
   a launch or reuse immutable weight/scale pointers under stable graph
   workspaces. Dynamic offsets and problem counts must still update on every
   replay. The C++ source itself marks descriptor setup as cacheable near line
   1378. Benchmark both prefill (priority) and captured small-M routes, including
   zero-active experts and -1 tails. Accept only a measured glue reduction with
   unchanged real-row outputs; this does not reduce the expert-byte floor.
2. **Conditionally mask unused BF16 draft routes in V2.** If raw capture confirms
   material unique-expert traffic, add a persistent per-forward draft mask in
   the vLLM speculator using GPU metadata. Start with graph-tail rows, then
   separately prove which rejected-suffix rows are unnecessary. Retain attention
   and KV semantics; do not compact/drop model input rows without proving their
   dependencies. Check token IDs, acceptance, draft/target outputs, graph replay,
   and end-to-end latency. This is a distinct draft optimization, not a claimed
   NVFP4 decode or prefill win.

Active-expert compaction can reduce sparse scheduler metadata traversal, but
CUTLASS currently receives a host-side group count. GPU-produced active counts
must not introduce a host synchronization or invalidate fixed CUDA graphs.
Without measured scheduler cost it ranks below these two candidates. Reordering
experts, shrinking MMA tiles, or pinning selected experts across tokens is not
evidence of lower required weight traffic; the last changes the model itself.
