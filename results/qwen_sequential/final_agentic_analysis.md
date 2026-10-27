# Final composed agentic result: modest, mixed gains

The selected scalar composition completes the final agentic comparison against
`next`, with **5,772 verified turns and 987,945 output tokens per configuration**
across all eight replay shapes. Both runs exit 0 and pass the strict comparison.
Observed output rates improve 0.34–3.92% in seven shapes and decline 3.60% in
short c=4. One sequential pair does not establish a robust or general speedup.

Keep scalar as an opt-in measured composition; default `next` remains unchanged.
The stronger cold-prefill gains from the earlier serving matrix remain separate
results. The additional GDN norm/FP8 overlay remains unselected for this current
configuration, following its full/focused incremental tradeoffs. See the
[exact patch selection](final_stack_selection.md).

## Eight measured replay shapes

Baseline: `next_final_agentic_1`. Candidate: `stack_final_agentic_1` (`scalar`).
Arrows show baseline → candidate. TTFT is over all turns; output rates count
verified completed outputs. Each row is one measured run per configuration.

| replay / concurrency | TTFT p50 ms | TTFT p90 ms | output tok/s | output change | wall seconds |
|---|---:|---:|---:|---:|---:|
| short c=1 | 104.79 → 98.24 | 139.82 → 130.67 | 331.09 → 344.08 | +3.92% | 98.94 → 95.20 |
| short c=4 | 126.55 → 121.12 | 191.09 → 205.25 | 765.83 → 738.27 | -3.60% | 42.77 → 44.37 |
| short c=8 | 126.34 → 127.11 | 225.66 → 198.37 | 955.20 → 978.61 | +2.45% | 34.29 → 33.47 |
| short c=16 | 192.97 → 180.97 | 369.30 → 335.26 | 1293.70 → 1318.91 | +1.95% | 50.64 → 49.67 |
| short c=32 | 257.85 → 253.59 | 545.95 → 548.03 | 1614.78 → 1650.00 | +2.18% | 81.14 → 79.41 |
| long c=1 | 250.66 → 249.81 | 343.02 → 340.77 | 248.32 → 249.18 | +0.34% | 465.21 → 463.61 |
| long c=8 | 336.45 → 335.63 | 698.23 → 775.53 | 440.05 → 444.11 | +0.92% | 262.52 → 260.12 |
| long c=32 | 866.08 → 816.42 | 1761.78 → 1757.58 | 557.81 → 571.79 | +2.51% | 828.40 → 808.15 |

At long c=32, TTFT p50 falls 866.08→816.42 ms (-5.73%) and wall time falls
828.40→808.15 s (-2.44%). Short c=4 wall time rises 42.77→44.37 s (+3.73%).
Long c=8 throughput improves only 0.92%, while TTFT p90 worsens
698.23→775.53 ms. These regressions remain part of the result; the pair does
not support a blanket latency or throughput claim.

The [full comparison](final_agentic_pair.md) and [machine-readable report](final_agentic_pair.json)
retain counts, later-turn timings, reported prefix-hit rates and set-level MTP
counters. The rate changes are +3.92%/-3.60%/+2.45%/+1.95%/+2.18% for short
c=1/4/8/16/32, and +0.34%/+0.92%/+2.51% for long c=1/8/32.

## Identity and runtime evidence

Both fresh servers use Qwen3.6-35B-A3B-NVFP4-Fast, the pinned vLLM 0.32.0 image,
TP1, MTP2, GPU budget 0.78 and the same `agentic` suite with greedy-decode and
NLL side suites disabled. The baseline's 21 captured source files and scalar's
23 match their manifests, and cleanup source verification passes. The only
extra runtime source paths are the scalar MoE backend and its patched oracle.
The existing `next` GDN path is retained in both; neither uses the additional
norm/FP8 overlay or its extra compilation split.

Both fixed-reference FlashInfer startup and shutdown checks verify all 132
entries, with effective-entry SHA256
`721ed789107ca7cd8d54b4ce8e6345a9e84e8882beca5914a4a2085598f1b063`.
The reference file SHA256 is
`7b5146f9b3c1e7cc09586c2087988ea8aaf9da1d86da036539a32dec4b335ec3`.
No serving lookup observer is installed. Inductor remains observed, not pinned:
three of 13 saved tuning records differ, with no added or removed records.
Those records do not identify every final per-launch choice or prove a compiler
cause for the measured differences.

The two input manifests are identical, and both captured files and live input
sources pass post-run verification. Compressed input SHA256 values are:

- Short: `e24df8aa498518add17b8188ef834ea4272ca78991513f6bbe0118f38021c680`.
- Long: `3852ffe3133be6ad2c93442682906f45990944f86e4126ff10f9c7cbf83580d8`.

The comparator validates metric version 2, all expected task/turn identities,
input/output lengths, completed streams and recomputed summary values. TTFT
begins at the first nonempty token/text payload. The reports do not independently
echo prompt-token hashes or generated cache salts; input identity is bound by
the captured files and source verification under the recorded runner protocol.

| startup allocation | next | scalar |
|---|---:|---:|
| available KV GiB | 44.61 | 45.50 |
| KV token capacity | 3,369,930 | 3,437,190 |
| final CUDA graph pool GiB | 4.90 | 3.45 |

The GPU budget matches; actual startup KV capacities differ. These are rounded
allocation logs, not a peak-memory benchmark or an isolated memory saving from
one patch. Reported prefix hits match: 68.84% for short, 94.55% for long c=1/8,
and 94.01% for long c=32. Raw counters per concurrency are absent, so those
fractions cannot be independently recomputed from the replay reports.

MTP acceptance is 88.64%→88.70% for the complete short set and
90.06%→90.39% for the complete long set. These are before/after counter deltas
over each whole set; they cannot explain a specific concurrency's result alone.

Evidence directories: [baseline](next_final_agentic_1) and
[scalar](stack_final_agentic_1), including source/input manifests, boundary
checks, request records, startup logs and graceful-stop logs. The historical
`processes_after.csv` samples are taken before cleanup and include the owned
server; they are not evidence of an empty GPU after shutdown. The separate
[post-run cleanup observer](final_agentic_cleanup.json), captured at
2026-10-27 23:04:44 UTC, records both owned containers absent and no GPU compute
processes, with hashes binding the final comparison reports.

## Separate live-short smoke

This small smoke generated fresh tool-calling trajectories after replay. It
is separate from the fixed replay and does not establish coding correctness.

| recorded smoke statistic | next | scalar |
|---|---:|---:|
| tasks | 8 | 8 |
| turns | 126 | 128 |
| tool calls | 126 | 132 |
| tool-result errors | 6 | 4 |
| malformed tool-argument calls | 0 | 0 |
| truncated turns | 0 | 0 |
| turns with no tool call | 0 | 0 |
| tasks calling `finish` | 1/8 | 1/8 |
| tasks not finished within the 16-turn cap | 7/8 | 7/8 |

The baseline errors are five missing-path operations and one `write_file({})`
missing its content argument. Scalar records four missing-path operations.
The fewer errors do not establish a quality improvement: trajectories differ,
only one task calls `finish` on either side, and the harness's `run_tests`
returns canned failure/pass responses without executing model-written code.
A `finish` call is a tool-use event, not independent proof that a task was solved.

The preserved artifacts are [baseline smoke](next_final_agentic_1/agentic_live_short.json)
and [scalar smoke](stack_final_agentic_1/agentic_live_short.json). Their wall
measurements, 96.40 s and 72.96 s, cover different generated work and are not a
paired speedup metric. The final replay's exit 0 does not convert these limited
smoke outcomes into a quality pass.

## Disposition and remaining limits

The requested sequence is complete: incremental correctness/micro/serving
experiments, measured stack selection, then one final whole-stack agentic pair.
Retain root `next` patches 01–05 plus MoE 03–06 as the opt-in scalar composition.
Keep the extra GDN norm/FP8 fusion available for scoped experiments. Default
`next` is unchanged; this single mixed pair does not justify a broad replacement
recommendation or a confident speedup claim.

The earlier scalar-versus-exact `py_email-30000` NLL flag remains: mean
0.794235→0.802070, with nonoverlapping three-repeat ranges. The GDN stack's
earlier `licenses-30000` flag versus `next` also stays recorded with its original
scope; GDN was not included in this final replay. Neither replay nor the live
smoke reruns or resolves those NLL diagnostics. Kernel correctness, small
needle/NLL probes and task counts do not establish full model-quality equivalence.

This is one sequential configuration pair without a confidence interval. It
fixes prerecorded prompts and output lengths, not generated-text equality or
live agent success. Inductor, MTP trajectories, scheduling and allocation
context remain interpretation limits. The earlier scalar cold 8k/16k/32k TTFT
gains of 18.8%/9.4%/7.1% versus `next` should not be presented as agentic gains.
