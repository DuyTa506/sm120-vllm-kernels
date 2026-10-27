# c1 routing validation checkpoints, 2026-10-28 local time

The latest fixed-reference native-aux/full results are at the end of this file.
Earlier failures remain below with their original scope.

## Initial independently tuned checkpoint

Exact output equivalence failed. Target-only capture completed inference and
saved structurally valid signed routes, but it is **not yet validated as
output-preserving**. No full draft capture or optimization adoption followed.
Both disposable containers were removed, both own-engine shutdown calls
completed, and the GPU was released to the root's sequential queue.
Production services were not changed. No full agentic benchmark ran.

## Preserved runs

| Run | Exit | Output | Capture |
|---|---:|---|---|
| `results/review_control_c1` | 0 | 192 tokens | None |
| `results/review_target_c1` | 1 | 192 tokens | First 64 pure-decode steps, target only |

Both use one existing recorded prompt (task 0, turn 0, 1,156 prompt tokens),
seed 101, temperature zero, MTP-2, CONFIGnext, V2, and CUTLASS NVFP4. Prompt SHA256:
`7184757dc71fd0d1ce9f3159f301d09af732a5cbc58386a9fac37f93f549c5e2`.

Reproduction of the completed jobs, using fresh tags if rerunning:

```bash
sg docker -c 'bash kernels/decode-route-capture/run.sh review_control_c1 --control --concurrency 1'
sg docker -c 'bash kernels/decode-route-capture/run.sh review_target_c1 --target-only --concurrency 1 --compare-output /work/kernels/decode-route-capture/results/review_control_c1/outputs.json'
```

Each directory retains `invocation.txt`, `exit_code.txt`, `run.log`,
`outputs.json`, `manifest.json`, `shutdown.json`, GPU/process snapshots, and image,
probe-source and CONFIGnext hashes. The target directory also retains
`raw_routes.pt`, `records.jsonl.gz` and `analysis_diagnostic.json`.
`results/review_comparison.json` records the CPU comparison and all tactic
differences. The original control and failed target artifacts were preserved.
Compiler caches and raw local artifacts are excluded by this directory's
`.gitignore`; these scripts and this summary are available for a root-owned commit.

## What failed, and what is still unknown

The first generated token differs: control ID 40, target-only ID 248058.
Overall 180/192 token positions differ. Prompt hashes, image ID, probe-source
hashes and CONFIGnext overlay hashes are identical. The logged engine
configurations are identical after normalizing only
`enable_return_routed_experts=False/True`.

Fresh independent autotuning selected different tactics in **51/132** entries:
21 `CutlassFp4GemmRunner` and 30 `MoERunner`. The autotune entry keys and version/
device metadata match. For example, the FP4 GEMM key beginning with input shape
`(1, 1024)` selected tactic 4 for the control and 0 for target capture. The
`_generation` field also differs and is excluded from the 132-entry comparison.
This is a concrete confound; it does not establish why the outputs differ.

The root's earlier repeated baseline quality runs also found greedy-continuation
variation. That context means the present mismatch cannot be attributed to
instrumentation alone. Conversely, it cannot establish that instrumentation
preserves outputs. The comparator correctly retained a failed status.

## CPU integrity checks and diagnostic routing observations

The authoritative `.pt` file was loaded with `map_location="cpu"` and CUDA hidden
from the analysis process. All 64 route tensors are signed `torch.int32` with
shape `[3, 40, 8]`. IDs, masks, input IDs, positions and GPU query boundaries
agree exactly with the JSON view. CPU/GPU request boundaries agree. All top-k
selections are valid distinct IDs in `[0, 255]`; acceptance counts satisfy the
pure-decode query-width invariant.

There are **zero graph-padding rows in this c1 sample**: each captured target
step has three real verifier rows. Thus this run provides no target graph-tail
masking observation and cannot support a target padding-waste claim. It does
include rejected verifier rows without filtering them. This result is scoped to
this one prompt/concurrency/configuration; it says nothing about larger batches
or the unmeasured draft's padding.

| Quantity | Diagnostic observation |
|---|---:|
| Target steps / layer groups | 64 / 2,560 |
| Scheduled draft tokens | 128 |
| Accepted / rejected draft tokens | 103 / 25 |
| Acceptance fraction | 80.46875% |
| Target sampler output tokens in captured steps | 167 |
| Mean unique experts per layer and three-row MTP group | 18.77109 |
| Mean adjacent-row expert intersection | 2.16973 |
| Mean first/third-row expert intersection | 1.83398 |

Live target geometry is H=2048, I=512, E=256, top-k=8, 40 layers. Loaded weight
and block-scale tensor sizes give **1,769,472 bytes per NVFP4 expert**. Summing
each captured step/layer's unique expert payload gives 85,030,207,488 bytes,
or 509,162,919 bytes per recorded sampler output token. Rejected verifier rows
remain included. The count covers the first 64 decode steps, not the entire
192-token output.

These are unique weight-plus-block-scale payload counts, **not measured DRAM
traffic, measured bandwidth floors, latency or speedups**. They exclude shared
experts, dense layers, attention and activations; actual memory traffic also
depends on cache residency and repeated tiles. Because output equivalence is
unresolved, use these observations only to inspect the capture and design the
next controls. Draft data is absent, not inferred from target expert IDs.

## Checks proposed at the initial checkpoint

1. Repeat the plain c1 control against the preserved original control with the
   same prompt, seed and output length; retain the result even if it differs.
   Compare autotune choices as well as output IDs.
2. Run the subsequently prepared `--native-aux-control`: enable
   `aux_output_config={"enable_return_routed_experts": True}` while omitting this
   probe's hook installation and raw-route cloning. At this initial checkpoint, the mode was CPU-tested but
   GPU-unrun. It separates the native aux path from this probe's extra
   snapshots. Repeat it to establish its own variability.
3. If independent autotuning remains a confound, arrange paired runs with
   verified identical autotune choices while keeping their compiled graphs
   separate. Compare native-aux-only control to target capture before adding
   draft capture. The optional importer and its verification are described in
   [CONTROLS.md](CONTROLS.md). No such paired run has occurred yet.

At the initial checkpoint, full target/draft capture was pending a new exclusive
GPU slot and useful controls. The later separately scheduled runs are below.

## Later checkpoint: fixed-reference plain control

`results/fixed_plain_c1` completed the separately authorized single GPU job with
exit code **0**. It used the original c1 control table through early preload and
fixed-reference replay, generated one 192-token output with `finish_reason=length`,
and completed owned engine and worker shutdown. This fresh reference run did
not request output comparison and captured no routes.

All six tactic stages—`preloaded`, `loaded`, `after_warmup`, `offline_startup`,
`after_generation`, and `offline_shutdown`—contained exactly 132 reference
entries, with zero missing, changed or extra entries. The saved source SHA256 is
`1dcdda546d5286e95e0a665a5aa89ef89f7d72dba84de132873dee38c4ecc991`; the canonical
effective entries SHA256 is
`8813376c0c2c4553e0f832c0629f720515b47689634f4362e23586de609254d8`.
The observer recorded 30,028 cache lookups covering 114 distinct keys, with zero
misses, uncovered hits or mismatched choices. This is observed-lookup coverage;
it does not imply that all 132 reference keys were executed.

The independent CPU lifecycle verifier passed. All 15 snapshotted probe source
files were unchanged at completion. Cleanup telemetry and a fresh check showed
no GPU compute processes, and inspection confirmed the owned container was
absent. The GPU slot was explicitly released before further CPU documentation.
`validation_summary.json` records the condensed evidence next to the original
outputs, stage snapshots, lookup report, invocation and telemetry.

At this intermediate checkpoint, only the plain control had run. Native-aux
and full target/draft capture were subsequently scheduled as recorded below;
a same-mode plain repeat has not run. The earlier target-only exact
mismatch remains unresolved; this successful control does not validate those
probe outputs or support optimization adoption. No latency or throughput claim
is made from this instrumentation run.

## Fixed-reference native-aux and full capture

Both later jobs completed generation, all FlashInfer lifecycle checks and owned
shutdown, but **failed exact output comparison**. The original failed manifests,
exit codes and diagnostic raw artifacts are preserved. The source/environment
and prompt hashes match the fixed plain control; all 15 recorded probe sources
and four CONFIGnext overlay sources still matched their launch hashes during
the independent CPU audit. No model runtime source was changed for this audit.

| Run | Exit | Exact comparison | Structural capture |
|---|---:|---|---|
| [`fixed_plain_c1`](results/fixed_plain_c1/validation_summary.json) | 0 | Not requested | No capture |
| [`fixed_aux_c1`](results/fixed_aux_c1/validation_summary.json) | 1 | Versus plain: 157/192 positions differ; first index 28 | Native aux only; no custom hooks or raw cloning |
| [`fixed_full_c1`](results/fixed_full_c1/validation_summary.json) | 1 | Versus native aux: 173/192 positions differ; first index 10 | Target and both draft phases, 64 steps each; structure passes |

All outputs finish at the requested 192 tokens. Every run has the same 132
reference tactics at all six boundaries and 30,028 observed cache lookups over
114 distinct keys, with zero misses, uncovered hits or mismatched choices.
Independent `offline_autotune.py` verification passes for both failed runs.
Engine and worker shutdown complete; each recorded post-cleanup process list
is empty. Parent cleanup also confirmed the owned containers absent before the
next GPU job. This does not change either exit code to success.

The full-run `baseline_variation` object compares plain against native aux and
reports distance 157; its full-versus-aux distance 173 is outside that supplied
envelope. These are different instrumentation modes, **not repeated samples of
one baseline configuration**. No fixed-reference same-mode repeat has been run,
so this is neither a variability estimate nor a statistical tolerance.

Observed `.best_config` files also differ: 5/13 settings between plain and aux,
5/13 between aux and full, and 6/13 between plain and full; no files are added or
removed by basename. These saved selections do not prove the final in-memory
choice of every runtime kernel. They are an unresolved compiler confound, not
an explanation established by this comparison. Native aux already differs
without the custom hooks, while the custom full capture also differs from aux;
neither observation establishes causality or validates output preservation.

### What the full signed capture shows

[`comparison_context_cpu.json`](results/fixed_full_c1/comparison_context_cpu.json)
records the condensed analysis, input hashes and compiler-file differences.
The independent standard-library analyzer reproduces the saved diagnostic
analysis exactly. A separate CPU-only Torch check loads `raw_routes.pt` with
`map_location="cpu", weights_only=True`, verifies every tensor and metadata value
against `records.jsonl.gz`, and confirms CUDA was never initialized; see
[`raw_integrity_cpu.json`](results/fixed_full_c1/raw_integrity_cpu.json) and its
adjacent reproducible script.

| Phase | Records | Signed route tensor per record | Required / rejected / graph-tail rows, summed over steps | Actual router mask |
|---|---:|---|---:|---|
| Target verifier | 64 | int32 `[3,40,8]` | 168 / 24 / 0 | Present, zero masked rows |
| First draft | 64 | int32 `[3,1,8]` | 168 / 24 / 0 | Absent |
| Second draft | 64 | int32 `[1,1,8]` | 64 / 0 / 0 | Absent |

All 192 records have valid distinct top-k IDs, and the raw values match their
JSON view. Target CPU/GPU query boundaries and proposal/final acceptance counts
agree. No draft phase is missing. The captured target steps propose 128 draft
tokens, accept 104 and reject 24: 81.25% acceptance and 168 sampler output tokens.
These counts cover the first 64 captured decode steps, not all 192 client tokens.

The target has 40 NVFP4 expert layers; the draft has one BF16 expert layer. Both
have H=2048, I=512, E=256 and top-k=8. Stored payload per expert is 1,769,472
bytes for target weights/block scales and 6,291,456 bytes for draft weights.
Mean target expert union over the three verifier rows is 18.72539 per layer;
mean adjacent-row and first/third-row intersections are 2.19395 and 1.83320.

| Phase | Sum of per-step/layer unique expert payloads |
|---|---:|
| Target verifier | 84,823,179,264 bytes |
| First draft | 7,933,526,016 bytes |
| Second draft | 3,221,225,472 bytes |

First-draft rejected suffix rows introduce 140 expert selections beyond the
required-row expert unions, summed independently over steps. Their associated
BF16 payload count is 880,803,840 bytes. This suggests a possible future check
of dead draft-row masking that preserves the required experts. It does not
prove those rows can safely be skipped across the complete draft path, nor
that these bytes reach DRAM or would translate into latency savings. There is
**no observed graph-padding waste** in this sample, and rejected target rows
must still be verified; they are not removable work on this evidence.

The payload counts do not measure traffic, bandwidth utilization, cache reuse,
latency, or speedup. They exclude shared experts, dense projections, attention
and activations. No route optimization is adopted or included in the selected
patch stack. Before developing one from this instrumented trace, a useful
follow-up would establish same-mode fixed-reference repeatability and isolate
remaining compiler/native-aux differences; it need not delay the independent
prefill stack's final serving/agentic evaluation.
