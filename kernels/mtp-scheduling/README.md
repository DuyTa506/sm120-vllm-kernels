# Existing MTP batch-dependent scheduling screen

Status: implemented, CPU checks passed; not GPU-tested and no gain claimed. The experiment compares fixed MTP2 with the installed vLLM schedule `[[1,8,2],[9,64,1]]`, maximum depth2 in both. This is a test boundary, not a tuned recommendation. No pressure-aware scheduler patch is installed.

`run.sh` is an isolated derivative of the repository sequential launcher, fixed to scalar + CONFIG=next and FI_CONTROL=1. It retains the pinned FI reference import, source-manifest verification, same model/backend/patch flags, GPU lock, unique container, cleanup and FI shutdown verification. Each run has a fresh compilation-cache namespace. The default speculative flag in serve.sh is suppressed using NOSPEC=1 only to avoid duplicate command-line configs; the launcher explicitly supplies its MTP config in both variants. Neither variant is a no-speculation baseline. Depth0 is not supported here and would not remove initial draft-prefill cost anyway.

The mount-only general plugin observes actual SchedulerOutput depth, request count and token count using CPU values. It also checks the active runner is the baseline V2 class and that the actual proposer accepts dynamic `num_speculative_tokens`; it writes class/propose source hashes and effective configuration. Missing or unsupported runtime hooks fail startup; no runner switch is performed. `MTP_DEPTH_PROBE=1` (default) is an instrumented diagnostic with synchronous line-buffered host logging. After both profiles pass runtime eligibility and actual depth checks, repeat paired timing with `MTP_DEPTH_PROBE=0`; this disables per-step observation while retaining startup class/config checks. Do not use instrumented timings as the final benefit estimate. A scheduled depth is the next proposal allowance, not the number of accepted tokens or necessarily the current verifier query length.

The primary stdlib streaming workload uses actual repository code tasks from `bench/agentic_bench.py` (literal task text only; no agent/tools), QA prompts from `eval/quality_probe.py`, and the long document from `kernels/moe-nvfp4/data/prompt.txt`. It tokenizes before timing, stores deterministic source/input hashes, starts16 code/QA requests with maximum1024 output, then adds two32768-token document requests with maximum128 output after each decoder emits text. Natural EOS is allowed; actual token counts are reported. It verifies both long-prefill arrivals overlap active decode. If natural completions end before overlap, reject the trial and adjust arrival scheduling identically for both variants rather than forcing artificial continuation.

`--synthetic-stress` explicitly retains the earlier random-token128/1024 input shape and ignore-EOS mode as a stress diagnostic only; it cannot support realistic acceptance claims. The raw stream timeline supports TTFT and gap analysis; SSE chunks can contain multiple accepted tokens, so chunk gaps MUST NOT be called token ITL. No agentic task is executed.

Prometheus snapshots record `vllm:spec_decode_num_drafts_total`, `vllm:spec_decode_num_draft_tokens_total`, `vllm:spec_decode_num_accepted_tokens_total`, and accepted-per-position counters. Compute deltas over the same run: accepted/draft tokens and accepted/draft calls. Do not interpret cumulative process totals as workload-only acceptance. Depth JSONL provides actual scheduled policy, not inference from startup configuration. Runtime trace currently includes startup and measured time; filter by request monotonic timestamps for workload analysis.

Root scheduler launch, only after current queued experiments:

```bash
sg docker -c 'bash kernels/mtp-scheduling/run.sh fixed2_mixed_1 fixed2'
sg docker -c 'bash kernels/mtp-scheduling/run.sh batch21_mixed_1 batch21'
```

Repeat in reverse order with new tags and identical seeds to assess noise. `DECODE_CONCURRENCY=4` provides a low-concurrency control where this schedule should retain depth2. New `SEED` values provide held-out prompt trials; reuse the same seed within each paired comparison. Before any promotion add repository task-level quality checks and a controlled steady-state pure-decode trial. An initial smoke can fail if all decode completes before the long-prefill arrivals; that is an invalid mixed-load trial, not a performance result.

Outputs are `kernels/mtp-scheduling/results/TAG`: source manifests, executable source archives and immutable experiment-source hashes (verified in cleanup even on failure), image/container/startup controls, effective runtime classes/config, host depth trace, FI verification, GPU monitoring, raw requests/metrics and exit code. No shared harness or serving source is edited.

CPU checks:

```bash
python3 -m unittest discover -s kernels/mtp-scheduling -v
bash -n kernels/mtp-scheduling/run.sh
```

The installed dynamic lookup was executed on CPU and confirmed requests1..8→2 and9..64→1; CUDA remained uninitialized. Source availability alone is not evidence that the live runner is correct: startup runtime guards remain required.

Timing-only repeat after eligibility diagnostics: prefix each launch with `MTP_DEPTH_PROBE=0`. Keep the same prompt manifest hashes and runtime class/source manifests across the pair. Cleanup verifies the owned container is absent and records the GPU compute list; cleanup failures produce a nonzero result.
