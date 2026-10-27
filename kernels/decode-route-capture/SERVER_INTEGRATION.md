# Fixed FlashInfer table for sequential serving comparisons

The optional mount-only plugin imports one immutable FlashInfer v1 JSON table
before the actual worker loads the model. It replays that table through native
non-tuning cache lookup during warmup and exports the effective table after graph
capture and before worker shutdown. It adds no routing hooks, raw tensor
copies, or per-lookup Python observer to serving. Default launches remain unchanged.
CPU validation passes. The first parent-owned GPU attempt failed before measured
requests because native cold-L2 tuning ignored the imported dense FP4 entries.
The revised replay mode is GPU-unvalidated; see the failure diagnosis below.

## Integrated launch

The parent-owned `bench/qwen_experiment.sh` now accepts `FI_CONTROL=1`. For example,
these are separate jobs for the root's exclusive sequential GPU queue:

```bash
sg docker -c 'FI_CONTROL=1 GREEDY_DECODE=1 NLL_ROUNDS=3 bash bench/qwen_experiment.sh scalar_matched_1 scalar individual'
sg docker -c 'FI_CONTROL=1 GREEDY_DECODE=1 NLL_ROUNDS=3 bash bench/qwen_experiment.sh exact_matched_1 shared_exact individual'
```

The runner owns the GPU lock and rejects an occupied GPU. It snapshots the six
runtime files before launch, gates before the first measured request, gracefully
stops its own container, and verifies the final table before removal. Fresh tags
are required. Do not nest another acquisition of the same lock around it.

The plugin is registered through `vllm.general_plugins` using the included
`.dist-info/entry_points.txt`. Mounting the directory on `PYTHONPATH` makes the
entry point available in both the API process and spawned workers. Registration
only installs lazy callbacks; the worker that actually loads the model creates
the importer and artifacts. An existing `VLLM_PLUGINS` allowlist must include
`qwen_fi_control`; preserve any other required plugins.

For another launcher, the equivalent container options are:

```bash
-v "$D/kernels/decode-route-capture:/opt/qwen-fi-control:ro"
-e PYTHONPATH=/opt/qwen-fi-control
-e QWEN_FI_IMPORT_JSON=/opt/qwen-fi-control/references/next_cpu_audit_autotune.json
-e QWEN_FI_IMPORT_SHA256=7b5146f9b3c1e7cc09586c2087988ea8aaf9da1d86da036539a32dec4b335ec3
-e QWEN_FI_IMPORT_OUT=/bench-results/fi_control
-e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/bench-results/fi_control/cache/autotune_import
```

Here `$D` is the worktree and `/bench-results` is a writable per-run artifact
mount. Preserve any existing required `PYTHONPATH` entries when composing these
options. Set the cache variable in the container environment before vLLM caches
environment values. The output directory must be fresh. No package installation
or installed-source mutation is needed.

## Immutable reference and compatibility

`references/next_cpu_audit_autotune.json` freezes the `next` table embedded in
`results/qwen_sequential/prefill_runtime_cpu_audit.json`. That audit captured the
historical cache used around `next_tasks_1`; the original live `next` cache path
has since changed. The new reference is a reserialized immutable snapshot, not
a claim that the current live cache or the original file's raw bytes are unchanged.
Its provenance file records both hashes and the historical path.

| Identity | Value |
|---|---|
| Reference JSON SHA256 | `7b5146f9b3c1e7cc09586c2087988ea8aaf9da1d86da036539a32dec4b335ec3` |
| Canonical tactic entries SHA256 | `721ed789107ca7cd8d54b4ce8e6345a9e84e8882beca5914a4a2085598f1b063` |
| Tactic count | 132 |
| Compared historical table | `shared_exact`, corresponding to the exact serving run's cache |
| Metadata and key sets | Identical in the CPU audit |
| Different tactic choices | 45 of 132 |

Matching metadata and keys establish that this is a plausible common reference.
Runtime compatibility still requires native `load_configs()` to return true,
the loaded table to match, and effective winners to remain equal. The importer
does not bypass FlashInfer's tactic validity checks. It supports only the
installed vLLM 0.32 / FlashInfer 0.7.0.post1 single-GPU, legacy-v1 cache API;
unsupported winner partitions fail explicitly.

An early native load runs at entry to `Worker.load_model`, after device setup
and before model construction, profiling or graph capture. Existing FI winners,
file entries, or active tuning cause an explicit failure; they are never silently
cleared. At native warmup the importer also seeds and checks the actual rank file.
Within this controlled warmup only, both vLLM's outer and nested autotune contexts
use FI's supported `tune_mode=False`. Native lookup and tactic validity checks
remain active; the plugin does not force tactic IDs or invent profiling-policy
metadata. The original context-manager API is restored even on failure. The
startup snapshot is taken after `Worker.compile_or_warm_up_model` returns,
including graph capture.

The source JSON is copied with exclusive creation into the actual rank path
resolved by vLLM, including the configuration hash and `_dp0_rank0` suffix. Only
JSON is imported. No CUDA graphs, JIT binaries or Inductor settings are imported.
The runner may still use its existing per-variant compiler cache; this plugin
does not make those caches fresh or equal.

## Startup and shutdown evidence

The standalone gate is CPU-only and supports isolated Python execution. With
`O=results/qwen_sequential/TAG`, run:

```bash
python3 -I kernels/decode-route-capture/qwen_fi_control_plugin.py \
  "$O/serving/fi_control" \
  --source-sha256 7b5146f9b3c1e7cc09586c2087988ea8aaf9da1d86da036539a32dec4b335ec3 \
  --startup-only
```

After requests finish, the owning launcher runs `docker stop -t 45 "$NAME"`
and invokes the same gate without `--startup-only`, before removing the
container. A forced kill or missing shutdown snapshot fails final verification;
a successful startup gate alone is insufficient for a complete controlled run.

Artifacts under `serving/fi_control/autotune/` include:

- `source.json`: exact imported bytes, with the declared SHA256.
- `verification.json`: early preload and rank-load returns, actual rank path, stage checks, reference
  identity in `source_sha256`, installed-code hashes in `runtime_source_sha256`,
  `import_mode=fixed_reference_replay_v1`, and `server_checkpoint`.
- `server_startup_effective.json` and `server_shutdown_effective.json`: complete
  FI-loadable snapshots, with original environment metadata and effective
  in-memory winners overriding loaded entries.
- `compiler_config_audit.json`: available selected Inductor configurations and
  hashes, recorded for comparison and never pinned by this plugin.

The gate requires the replay mode, a successful preload at
`before_worker_load_model`, and the startup boundary
`after_compile_or_warm_up_model`, in addition to the appropriate verified checkpoint, matching declared and
saved-source hashes, and exact effective-entry hash equality including the key
set. `verify_artifacts(out, expected_sha256, require_shutdown=True)` returns
`verified_stage`, `source_sha256`, `effective_entries_sha256`, `entries`,
`import_mode`, `preload_boundary`,
`inductor_pinned=False`, and `serving_lookup_observer=False`.

A paired comparator should require both runs to declare `FI_CONTROL=1`, use the
same reference bytes and six source hashes, pass both startup and final gates,
and have equal effective hashes. Recompute the canonical reference entries hash
from `source.json` when independently auditing the report. Reject comparisons
between controlled and uncontrolled runs as matched-cache evidence.

These checks establish table identity at the recorded boundaries. Without the
offline probe's per-lookup observer, they do not prove that every individual
lookup avoided a fallback. They also do not pin Inductor/Triton choices or
establish deterministic tokens. Separately report compiler differences, exact
output equality, baseline variation, and performance. A token mismatch alone
does not identify instrumentation as the cause.

## Runtime source set and tests

Freeze and snapshot these six paths together during an active pair:

```text
kernels/decode-route-capture/qwen_fi_control_plugin.py
kernels/decode-route-capture/autotune_import.py
kernels/decode-route-capture/qwen_fi_control-0.1.dist-info/METADATA
kernels/decode-route-capture/qwen_fi_control-0.1.dist-info/entry_points.txt
kernels/decode-route-capture/references/next_cpu_audit_autotune.json
kernels/decode-route-capture/references/next_cpu_audit_autotune.provenance.json
```

The 47 CPU tests cover actual importer flow against mocked FI/vLLM dependencies,
source identity through both server gates, native method return preservation,
lazy worker ownership, missing shutdown, effective drift, rejected loads,
uncovered lookups, cleanup despite audit failure, entry-point discovery, and the
`python3 -I` CLI. They also reproduce cold-L2 retuning of a successfully loaded
legacy table, verify replay of both nested contexts, reject preexisting winners,
and check native API restoration after failure. No GPU or live HTTP requests are
made by these tests.

## Failed first import and the diagnosed cause

`results/qwen_sequential/scalar_matched_1` retains the failed first attempt.
The reference loaded with all 132 entries unchanged. After native warmup,
20 `CutlassFp4GemmRunner` entries differed, including M=1 tactic 7 becoming 0
and M=1024 tactic 13 becoming 12 for packed K=1024. The strict check aborted
startup; no measured request or paired performance comparison was completed.
These original artifacts and their failure status are unchanged.

The installed source explains the dense-only retuning:

- `flashinfer/gemm/gemm_base.py:7887-7890` configures 128x4 FP4 GEMM with
  `use_cuda_graph=True, use_cold_l2_cache=True`.
- `flashinfer/autotuner/autotuner.py:3304-3312` assigns legacy v1 persisted
  entries a hot-L2 default policy.
- The same file, `1969-1973`, deliberately skips file entries while tuning when
  the requested policy differs. Lines `2330-2337` then profile cache misses.
- `vllm/model_executor/warmup/kernel_warmup.py:483` opens the outer tuning context;
  line `374` opens a nested context with default tuning enabled.
- Saved logs show dense FP4 retuning, while MoE reports file-cache hits.

Consequently, loading earlier or clearing memory winners alone cannot prevent
this retuning. The revised control explicitly selects fixed-reference replay
instead of claiming that a v1 JSON records cold-L2 measurement provenance. The
old attempted mode is named `native_tuning_with_reference_v1`; it cannot pass the
new server gate. Verification has not been relaxed.

## Pending decode controls

The offline commands in [CONTROLS.md](CONTROLS.md) remain pending a separate root
GPU slot **and equivalent early fixed-reference replay integration in the offline
driver**. Its current optional importer retains native tuning, so dense FP4 can
hit the same cold-L2 override and correctly fail verification. Use the preserved original **decode control** JSON for those commands:
its warmed shapes correspond to that workload. Do not assume the serving table
above covers every decode/native-aux/draft lookup. The offline importer observes
actual lookup results and `--require-autotune-coverage` fails missing coverage.

Run a plain control repeat first, then native-aux-only control, inspect their
tactic and output evidence, and then run full target/draft capture with the same
reference. Repeats quantify baseline variation separately from the exact
comparison, which retains its failure status. If strict coverage fails, preserve
the artifact and determine which keys are missing before scheduling another
matched pair; do not silently treat a partial import as matched execution. Full
capture still requires structural validity of signed target/draft routes, actual
masks, rejected rows and MTP acceptance. The current c1 target checkpoint remains
an unresolved exact-equivalence failure and supports no optimization adoption.
