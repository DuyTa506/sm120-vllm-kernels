# Sequential Qwen SM120 experiments and final serving evaluation

User instruction: test each experiment in sequence, update README in the
repository's format, measure its effect in vLLM on the existing workload
shapes, then combine successful changes and run agentic workloads.

Target remains Qwen3.6-35B-A3B-NVFP4-Fast, vLLM 0.32.0, one RTX PRO 6000,
MTP-2, TP/EP=1. The existing `CONFIG=next` is the serving baseline. Work is
isolated in `perf/qwen-prefill-workspace`; do not change live services.
The user authorized sequential GPU experiments after Arm A training ended
at step 1240 and wrote its EMA checkpoint on 2026-10-27 at 23:49 local time.
The earlier GPU-priority question is superseded by this execution instruction.

## Ordering and evidence

- One GPU experiment/server at a time, using the repository GPU lock. CPU
  preparation/review can run in parallel; no concurrent GPU agents.
- Keep immutable runner snapshots for active jobs. Record source/config,
  image, seeds, memory budget, detailed samples and GPU telemetry.
- Correct cold-prefill protocol: disjoint warmup and measured prompts, unique
  seeds across lengths, CLI readiness/warmup disabled on measured calls.
  Shared-prefix is deliberately cache-friendly and is reported separately.
- Treat negative findings as completed experiments: document them and exclude
  them from the combined configuration. Do not force adoption to fill a table.
- Compare each change with its immediate unchanged baseline; retain `next`
  versus final-combination comparisons as a separate question.

## Experiments

1. **Shared b12x workspace** (`kernels/moe-nvfp4/shared/`).
   Allocation/lifetime tests and 110 numerical/replay checks passed. Full
   serving initialized all 40 layers. Shared-GPU A2/B1 is exploratory.
   Clean cold-v2 A3 CUTLASS then B2 shared completed (session 38206 exited 0).
   Record 8k/16k TTFT, nsys GPU span, KV capacity, c=1/c=4 decode. Repeat in
   reverse order if the gain is comparable to timing variation. Revalidate
   real-layer microbenchmarks as necessary for clean performance attribution.
2. **Scalar FC1 activation scale** (`kernels/moe-nvfp4/scalar/`).
   CPU preflight passed 24 cases. Next run explicit GPU correctness/dispatch
   and interleaved full-apply timing, with uniform and nonuniform scales,
   changing routing, graph replay and the 1023/1024 boundary. The baseline is
   shared b12x. Account for FlashInfer's added scalar broadcast copy. If useful,
   measure this incremental change in serving before including it.
3. **GDN RMSNorm + gate + dynamic token FP8 fusion**
   (`kernels/gdn-norm-fp8/`, preparation in progress).
   Preserve the BF16 rounding boundary and one scale per 4096-value token.
   Check contiguous/strided gates, edge cases, exact installed compiled
   reference and replay. Measure isolated complete chain, then unchanged
   projection GEMM consumer. Only integrate an opt-in guarded path after
   useful microbenchmark and numerical evidence, then measure serving.
4. **Decode routing investigation** (`DECODE_ROUTING_RESEARCH.md`).
   Capture raw int32 target/draft routes, valid/padded/rejected rows and MTP
   phases from the actual V2 runner. Estimate expert reuse/bytes from observed
   decode routes, not random prefill subsets or unmasked M=4 proxies. Target
   NVFP4 padding is already masked; any draft padding opportunity is BF16
   and must be measured separately. Preserve learned top-k choices. If no
   actionable gain is supported, record that result rather than changing routing.

## Serving task matrix for retained candidates

Use the existing workload shapes and request counts from
`bench/throughput_sweep.sh`, with its readiness reuse fixed and detailed output:

| task | input/output | concurrency | requests |
|---|---|---:|---:|
| long-doc | 32768/128 | 2 | 8 |
| prefill-heavy | 8192/64 | 8 | 48 |
| rag | 4096/256 | 16 | 96 |
| shared-prefix | 4096 shared + 1024/256 | 16 | 96 |
| mixed-len | 4096/512, range ratio 0.9 | 32 | 128 |
| chat | 1024/512 | 32 | 128 |
| decode-heavy | 128/1024 | 64 | 192 |

Also preserve the repository prefill length sweep (2048,4096,5000,8192,16384,
32768), decode c=1/4/16/64, and existing relevant model-quality checks
(needles, NLL/logprobs, checkpoint-state behavior, MTP acceptance). Summarize
task throughput, TTFT/TPOT/ITL and cache capacity; rerun material regressions
or small uncertain wins before adoption. Use a fixed equal memory budget for
each pair; initial clean A/B uses 0.60, broader serving/agentic uses 0.78 unless
measured capacity requires a justified common adjustment.

## Final combination and agentic gate

- User explicitly clarified: **agentic runs only at the final combination
  stage**, comparing baseline with the composed winners. Never run agentic
  for each individual patch. `qwen_experiment.sh individual` runs serving
  and quality only; `agentic` is a separate, explicitly selected final stage.
- Build one explicit opt-in configuration from independently supported wins.
  Verify patch application, compile-cache isolation and numerical checks again
  where composition could affect behavior. Run the serving matrix against next.
- Replay the **same production-recorded token trajectories** through next and
  the combined configuration using `bench/agentic_bench.py`: short sessions
  c=1/4/8/16/32 and long sessions c=1/8/32, as in the current README.
- Preserve cache salts and record turn failures, TTFT p50/p90, turns/s,
  output tokens/s, prefix hit rate, session wall time and MTP acceptance.
  Replay measures performance; retain an actual tool-calling smoke/quality
  check separately, without claiming replay establishes model quality.
- Update each kernel README and the root README in the existing table format,
  link raw results and commands, identify confirmed/rejected/unresolved results,
  and replace outdated unshared-b12x caveats with scoped current evidence.
  Publish no unmeasured combined or agentic gain.

Completion requires executed experiments, documented decisions, a measured
combined configuration, and the agentic comparison. Prepared scripts and
microbenchmarks alone do not satisfy this request.

## Progress log

- Clean b12x pair: 8k GPU span132.33→120.23 ms, two TTFT seed means lower at
  both 8k and16k. Decode short-run throughput is6.35% lower despite similar
  ITL; investigate with reversed-order full task matrix before adoption.
- Clean shared/unshared micro replay passed10 more numerical checks (120
  total); no sharing-induced slowdown. Clean layer19 CUTLASS/b12x integrated
  micro time887.0→654.6 us at8k, matching the previously verified kernel gain.
- CPU preparation complete: scalar overlay+24-case preflight, standalone GDN
  fusion+verifier, routing study. Their GPU experiments remain pending.
- Full matrix harness reviewed; fixed exact-output-length validation and
  cross-scenario prefix isolation. It reconstructs mixed output lengths using
  the installed CPU dataset sampler, records34 distinct phase salts and fails
  incomplete requests.28 saved A3/B2 result files pass its checks.
- Agentic harness reliability fixes pass7 mocked CPU tests. No agentic GPU
  run has occurred; it remains exclusively the final combined-config task.
