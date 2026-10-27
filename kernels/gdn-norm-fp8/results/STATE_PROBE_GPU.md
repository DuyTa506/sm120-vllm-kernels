# Packed GDN state parity: scoped pass

`packed_state_probe_3` exited0. Layer18 used actual eager MTP2 inputs/metadata
and baseline auto-MoE activations. The reference was the unchanged original
packed core plus explicitly compiled raw `QuantFP8.forward_native`. Candidate
`packed_norm_fp8` ran on independent private caches initialized from the same
actual state. All required coverage gates passed without relaxing comparisons.

| Checked case | Actual tokens | Route | Compared padded T | Accepted-token metadata |
|---|---:|---|---|---|
| Mixed speculative decode + prefill |2048 =2045 prefill+3 spec|fused|2048,2065|[1]|
| Pure speculative decode |6|original fallback|6,23|[3,1]|
| Pure prefill chunk from cold8192-token request |2048|fused|2048,2065|None|

All six variants have zero FP8-byte, scale-byte, convolution-cache-byte and
recurrent-cache-byte differences. Padding leaves actual Q/S and private
updated state unchanged; padded Q is zero and scales equal the floor.
Added padded QKVZ/BA rows contain NaNs. Full selected-layer live-cache bytes
were unchanged by every private comparison; the original model call then
advanced the live state exactly once.

The worker observed7 mixed-spec batches,14 pure-decode batches and4 long-prefill
batches over29 hooked calls. It checked one case of each required class, at
original T and T+17. The mixed gate explicitly required at least1024 actual
tokens and the fused route for both padding variants. Two initial requests
(64/8192 prompt tokens) each completed32 output tokens; the separate long-only
8192-token request used a distinct cache salt and completed one output token.

The owned engine shut down normally. The runner verified its random labelled
container absent; a separate inspect also found no object. A fresh compute
query was empty before the GPU slot was released. No force-removal was needed.

This is a layer18 eager/private-cache proof, not full-model paired quality,
CUDA-graph replay, async scheduler/allocator identity or serving performance.
No checkpoint-bearing metadata or mixed non-speculative decode case was
observed, so those are not claimed covered. Actual checkpoint scratch cloning
has CPU coverage only. The B12x compiled integration smoke and standalone
norm/quant/GEMM graph checks are separate evidence.

Earlier attempts remain preserved: probe1 failed during MTP construction
because the fixture forced an unsupported unquantized MoE backend; probe2 had
exact checked mixed/decode results but correctly failed missing pure-long
coverage. Probe3 retained the mixed workload and added a separate salted
long-only request. Its worker, numerical criteria and coverage gate were
unchanged from probe2. Neither earlier status is relabelled.

`packed_state_probe_3_summary.json` records metadata, counters, cleanup and
source hashes. `packed_state_probe_3_sources/` preserves every source listed
in the run's `api.probe_sources`, verified against its recorded digest.
`summarize_state_probe3_cpu.py` reproduces the saved-result assertions using
only the Python standard library; it does not access GPU or HTTP.
