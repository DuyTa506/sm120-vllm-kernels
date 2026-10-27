# c1 validation checkpoint, 2026-10-28 local time

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

## Next queued checks, not run at this checkpoint

1. Repeat the plain c1 control against the preserved original control with the
   same prompt, seed and output length; retain the result even if it differs.
   Compare autotune choices as well as output IDs.
2. Run the subsequently prepared `--native-aux-control`: enable
   `aux_output_config={"enable_return_routed_experts": True}` while omitting this
   probe's hook installation and raw-route cloning. This mode is CPU-tested but
   GPU-unrun. It separates the native aux path from this probe's extra
   snapshots. Repeat it to establish its own variability.
3. If independent autotuning remains a confound, arrange paired runs with
   verified identical autotune choices while keeping their compiled graphs
   separate. Compare native-aux-only control to target capture before adding
   draft capture. The optional importer and its verification are described in
   [CONTROLS.md](CONTROLS.md). No such paired run has occurred yet.

Full target/draft capture remains pending a new exclusive GPU slot and a useful
control comparison. No further GPU work was started after release.
