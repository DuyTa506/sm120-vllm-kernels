# Controls and verified FlashInfer JSON import

Prepared after the failed c1 checkpoint. The fixed-reference plain control
`fixed_plain_c1` subsequently passed its single authorized GPU run; native-aux
and full capture remain GPU-unrun under this reference. Default inference options
are unchanged. See [the recorded control evidence](RESULTS.md).

The offline driver now uses a separate `offline_autotune.py` lifecycle wrapper
for explicit early fixed-reference replay. It reuses the serving-proven importer
API without modifying the six serving runtime files. The offline composition is
CPU-tested and validated for the plain c1 control. See [the cold-L2 source diagnosis](SERVER_INTEGRATION.md).

## Options

- `--native-aux-control` enables vLLM's native aux connector but does not create
  `V2RouteCapture`, install its router/phase hooks, or clone its raw route tensors.
- `--autotune-json PATH` preloads one explicit saved FlashInfer JSON before the
  owned worker loads the model, then imports it into the new run's
  `cache/autotune_import/<actual-config-hash>/` rank file. Warmup explicitly
  replays native cache lookup with `tune_mode=False`, avoiding cold-L2 retuning
  of the legacy v1 table. No compiled
  graphs, FlashInfer JIT binaries or Triton tuning data are imported. The source
  is read-only and an existing destination is rejected.
- `--require-autotune-coverage` additionally fails if any observed FlashInfer
  lookup misses or uses a key absent from the imported table. Without it, partial
  coverage is explicitly reported. Missing or changed imported entries always fail.
- Repeatable `--baseline-output PATH`, alongside `--compare-output`, reports
  positional token distances for supplied same-prompt controls. The
  `within_observed_token_distance` field is descriptive, not equivalence or a
  statistical tolerance. Exact mismatch still produces exit 1.

## Import and verification

The offline wrapper requires the owned in-process engine and rejects combination
with the serving plugin. It preloads before model construction and refuses any
preexisting FI winners or file entries. The importer wraps the existing warmup boundary, seeds the actual rank file
resolved by vLLM, and observes the original `AutoTuner.load_configs()` return.
A rejected load fails instead of silently continuing with fresh tuning. It
compares the loaded table and effective winners after warmup, graph capture,
generation and before worker shutdown.
A read-only `search_cache` observer records choices returned while building
graphs or running inference and returns the original result unchanged. It
never forces unsupported tactics, replaces expert IDs or recomputes routing.

The driver verifies the startup artifacts before generation. It exports raw
routes and structural analysis before calling its owned engine's shutdown API.
The worker hook saves `offline_shutdown_effective.json` before native worker
cleanup, and the driver then verifies both shutdown completion and artifact
consistency. `autotune/startup_verification.json` and
`autotune/lifecycle_verification.json` record these independent gates. A final
tactic or cleanup failure marks `manifest.success=false` without replacing a
prior exact-output error or structural result. Constructor failures and partial
captures remain failures with their available artifacts preserved.

`autotune/verification.json` records source hash, source-code hashes, load calls,
missing/changed/extra entries, observed choices and coverage. Stage tactic tables
are also saved. Inspect these even when the run fails for token mismatch.
`reference_and_observed_tactics_verified` means observed FlashInfer cache
lookups were covered and matched. It does **not** establish global determinism:
separately compiled Triton/Inductor kernels and other numerical or scheduling
differences remain possible. CUDA graph replay executes previously chosen
kernels; the observer records their choices at graph construction, not a Python
callback on every replay.

`autotune/compiler_config_audit.json` records available Inductor `.best_config`
files, raw hashes and hashes of selected settings excluding timing/cache metadata.
These compiler settings are observed, never imported or pinned. Missing files
or audit errors are explicit; this is not a complete audit of every compiler.

Installed source evidence:

- `vllm/model_executor/warmup/flashinfer_autotune_cache.py:38-55` resolves the
  configured cache root and actual configuration hash.
- `vllm/model_executor/warmup/kernel_warmup.py:453-476,501-504` adds the rank suffix,
  loads configs and saves dirty tuning state.
- `flashinfer/autotuner/autotuner.py:1936-1996` gives in-memory winners priority
  over loaded JSON and validates cached tactics before returning hits.
- The same file, `3361-3429,3559-3695`, defines save precedence and load rejection.
  The probe reads the corresponding tables without calling `save_configs`, which
  would clear dirty state and merge disk entries.

The current scope is vLLM 0.32 / FlashInfer 0.7.0.post1, one GPU, legacy v1
FlashInfer tactic tables. Managed/v2 winner partitions and namespaced tables are
rejected. CPU tests exercise rejected loads, overwritten live winners, missing
coverage, output failure, source immutability and cache isolation. The real
importer plus offline lifecycle also passes against mocked cold-L2 dependencies
through every boundary. The plain c1 GPU run also passed all six boundaries;
the native-aux and route-capture compositions still require GPU validation.

## Proposed sequential jobs

Each command is a separate job for a **new exclusive GPU slot**. All retain the
192-token/64-step/c1 defaults. Fresh tags are required for reruns.

`fixed_plain_c1` below already exists and passed. Reuse its saved outputs for
the next jobs; do not rerun its tag. A control repeat must use a fresh tag.

```bash
export ROUTE_AUTOTUNE_JSON=/work/kernels/decode-route-capture/references/decode_control_autotune.json
sg docker -c 'bash kernels/decode-route-capture/run.sh fixed_plain_c1 --control --autotune-json "$ROUTE_AUTOTUNE_JSON" --require-autotune-coverage'
sg docker -c 'bash kernels/decode-route-capture/run.sh fixed_plain_repeat_c1 --control --autotune-json "$ROUTE_AUTOTUNE_JSON" --require-autotune-coverage --compare-output /work/kernels/decode-route-capture/results/fixed_plain_c1/outputs.json'
sg docker -c 'bash kernels/decode-route-capture/run.sh fixed_aux_c1 --native-aux-control --autotune-json "$ROUTE_AUTOTUNE_JSON" --require-autotune-coverage --compare-output /work/kernels/decode-route-capture/results/fixed_plain_c1/outputs.json --baseline-output /work/kernels/decode-route-capture/results/fixed_plain_repeat_c1/outputs.json'
sg docker -c 'bash kernels/decode-route-capture/run.sh fixed_full_c1 --autotune-json "$ROUTE_AUTOTUNE_JSON" --require-autotune-coverage --compare-output /work/kernels/decode-route-capture/results/fixed_aux_c1/outputs.json --baseline-output /work/kernels/decode-route-capture/results/fixed_plain_c1/outputs.json --baseline-output /work/kernels/decode-route-capture/results/fixed_plain_repeat_c1/outputs.json'
```

The immutable reference is a byte-for-byte copy of the preserved original
`review_control_c1` rank table. Its SHA256 is
`1dcdda546d5286e95e0a665a5aa89ef89f7d72dba84de132873dee38c4ecc991`, with 132
entries and canonical entries hash
`8813376c0c2c4553e0f832c0629f720515b47689634f4362e23586de609254d8`.
The adjacent provenance JSON records the original path. It is separate from
the serving reference and preserves the original route control's selected tactics.

The minimum path is plain control, native-aux control and full capture; the plain
repeat above additionally measures baseline variation. Each command is scheduled
and inspected separately, so an exact mismatch can be investigated before any
further diagnostic run. No command launches HTTP or the full agentic benchmark.

After a run, independently check its tactic lifecycle on CPU:

```bash
python3 -I kernels/decode-route-capture/offline_autotune.py \
  kernels/decode-route-capture/results/fixed_full_c1
```

Inspect each tactic audit and output comparison before scheduling the next job.
The original control was independently tuned; comparison against it is
historical, not proof of a repeat under identical runtime choices. Repeat
`fixed_plain_c1` or `fixed_aux_c1` with fresh tags to measure each control's
own variation. Pass those repeated outputs via `--baseline-output` when a
separate descriptive envelope is useful. Choose matching workload/configuration
controls; the token-distance calculation does not validate their configurations.

## Failure and structural status remain separate

Completed inference triggers structural route analysis even if exact output or
tactic comparison fails. The overall run remains failed; `structural_validation`
reports its own result, and structurally valid diagnostic analysis is saved as
`analysis_diagnostic.json`. Full capture still requires target and both draft
phases, signed routes, actual draft masks and acceptance consistency. Failed or
incomplete raw evidence is retained. Being within observed baseline token
distance never turns a failed exact comparison into a passed run.
