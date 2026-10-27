# Qwen NVFP4 prefill: share b12x scratch across layers

The user wants continued SM120 GEMM optimization for Qwen3.6-35B-A3B
NVFP4 in vLLM, prioritizing prefill while preserving decode. They authorized
continuing implementation and will stop the unrelated GPU training job.

## First deliverable

Make the existing scale-corrected hybrid b12x backend usable without allocating
one large scratch/output set per MoE layer. This is a prerequisite for measuring
the already observed MoE prefill speedup in a complete serving workload.
It is not evidence of a new kernel speedup by itself.

Target: one RTX PRO 6000 Blackwell (SM120), vLLM image v0.32.0,
`unsloth/Qwen3.6-35B-A3B-NVFP4-Fast`, TP=1, EP=1, MTP-2, CONFIG=next.
The local image ID is
`sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34`.
Its WorkspaceManager exposes `get_persistent_resource` and owner lanes.
FlashInfer exposes `shared_static_workspace`, `shared_dynamic_workspace`,
and `shared_output`. The implementation depends on these verified APIs.

## Alternatives and decision

1. Port DeepGEMM's MXFP4 kernels to NVFP4. This needs a different scale
   contract and profiling to identify transferable improvements; defer.
2. Write/fuse new CUTLASS prefill kernels. Potentially useful, but the existing
   b12x path already saves about 220–228 us per MoE call at 8k in the recorded
   microbenchmarks. Establish a usable baseline first.
3. Share the existing b12x workspaces and validate in vLLM. Selected as the
   first bounded change under the user's prefill priority.

## Behavior and lifetime

- Add opt-in `VLLM_B12X_SHARED_WORKSPACE=1` on top of the existing scale fix.
- Keep original files/patch 03 as the A/B baseline; ship a candidate full-file
  overlay plus incremental patch 04.
- Resolve scratch ownership through vLLM's current WorkspaceManager on every
  b12x invocation, respecting its ubatch/lane partitions and warmup lock.
- Identical shapes in the same execution owner share static/dynamic scratch
  and temporary output. Each layer retains its own wrapper and weight/scale
  view caches. Different configurations/owners use distinct buffers.
- Layers in one owner execute serially, as required by vLLM's workspace
  contract. Concurrent forward streams must use distinct execution owners.
- Keep output.copy_ so a later layer cannot overwrite a previous returned
  output. Do not replace it with a scratch alias in this change.
- Run the CUTLASS small-batch fallback before creating b12x wrappers. Preserve
  autotune behavior, routing, weights, quantization scales and threshold.
- Retain both workspace families initially: avoiding the unused static family
  is a separate optimization requiring a different FlashInfer allocation API.

## Validation

First reproduce per-layer GPU allocation growth. Then check bounded allocation
across identical layers; isolation across shapes, lanes and ubatches; manager
reset and lock behavior; and default-off behavior. Use actual FlashInfer
wrappers, with GPU kernels tested against the unchanged b12x path.

Replay two layers in CUDA graphs with changing routing and validate outputs,
including masked IDs and batches around the hybrid threshold. Run server A/B
with separate compile caches, fixed prompts, 8k/16k prefill and decode checks;
record KV capacity, GPU span, TTFT and MTP statistics. The user subsequently authorized measurements alongside the training workload.
Record GPU load and compare both timing orders; never stop that process ourselves.

Acceptance: scratch growth is per execution owner/configuration, not per layer;
no correctness failures; no measurable regression due to sharing. Do not claim
an end-to-end speedup until supported by repeated measurements, with shared-GPU noise stated explicitly. Generalized
model support, routing changes, new GEMM kernels and default enablement are
outside this first deliverable.
