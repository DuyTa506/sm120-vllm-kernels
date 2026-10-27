# Controls and verified FlashInfer JSON import

Prepared after the failed c1 checkpoint; CPU-tested and **GPU-unrun**. Default
inference options are unchanged. No additional GPU job was started.

## Options

- `--native-aux-control` enables vLLM's native aux connector but does not create
  `V2RouteCapture`, install its router/phase hooks, or clone its raw route tensors.
- `--autotune-json PATH` imports one explicit saved FlashInfer JSON into the new
  run's `cache/autotune_import/<actual-config-hash>/` rank file. No compiled
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

The importer wraps the existing warmup boundary, seeds the actual rank file
resolved by vLLM, and observes the original `AutoTuner.load_configs()` return.
A rejected load fails instead of silently continuing with fresh tuning. It
compares the loaded table and effective winners after warmup and generation.
A read-only `search_cache` observer records choices returned while building
graphs or running inference and returns the original result unchanged. It
never forces unsupported tactics, replaces expert IDs or recomputes routing.

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
coverage, output failure, source immutability and cache isolation. GPU runtime
compatibility of this importer is still untested.

## Proposed sequential jobs

Each command is a separate job for a **new exclusive GPU slot**. All retain the
192-token/64-step/c1 defaults. Fresh tags are required for reruns.

```bash
export ROUTE_AUTOTUNE_JSON=/work/kernels/decode-route-capture/results/review_control_c1/cache/vllm/flashinfer_autotune_cache/0.7.0.post1/120f/729147d8b9317888627ef7c3767144c675b0c1d08bc813807383c0d69263e11e/autotune_configs_dp0_rank0.json
sg docker -c 'bash kernels/decode-route-capture/run.sh matched_plain_c1 --control --autotune-json "$ROUTE_AUTOTUNE_JSON" --require-autotune-coverage --compare-output /work/kernels/decode-route-capture/results/review_control_c1/outputs.json'
sg docker -c 'bash kernels/decode-route-capture/run.sh matched_aux_c1 --native-aux-control --autotune-json "$ROUTE_AUTOTUNE_JSON" --require-autotune-coverage --compare-output /work/kernels/decode-route-capture/results/matched_plain_c1/outputs.json'
sg docker -c 'bash kernels/decode-route-capture/run.sh matched_full_c1 --autotune-json "$ROUTE_AUTOTUNE_JSON" --require-autotune-coverage --compare-output /work/kernels/decode-route-capture/results/matched_aux_c1/outputs.json'
```

Inspect each tactic audit and output comparison before scheduling the next job.
The original control was independently tuned; comparison against it is
historical, not proof of a repeat under identical runtime choices. Repeat
`matched_plain_c1` or `matched_aux_c1` with fresh tags to measure each control's
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
