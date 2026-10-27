# Final agentic replay comparison

`compare_agentic.py` compares completed `qwen_experiment.sh ... agentic` artifacts offline. It requires metric version 2 and the fixed production trajectories. It shares the strict container/config/source/GDN/FI gates with `compare_runtime_identity.py`; the prior greedy-decode/NLL settings must also match. It checks every `(copied task ID, turn)` exactly once, its input length and requested/verified completion count, successful finished streams, finite per-turn timings, recomputed summary counts/rates/percentiles, and monotonic MTP counters.

The input snapshot and verification steps below are already wired into the runner, before the matched-3 individual experiments. Only the explicit `agentic` suite invokes them. Existing replay reports without metric version 2 or captured input evidence are rejected. The final `next_final_agentic_1` / `stack_final_agentic_1` (`scalar`) pair now completes exit 0 on both sides, with 5,772 verified turns and 987,945 output tokens each. The [strict comparison](../results/qwen_sequential/final_agentic_pair.md) and [analysis](../results/qwen_sequential/final_agentic_analysis.md) record modest, mixed effects and limited live-smoke results. Individual suites never run agentic.

## Current runner integration

`bench/source_manifest.py` already includes `bench/agentic_input_manifest.py` in its common source list. Keep the same runner/helper versions for the baseline and candidate. The existing `SUITE == agentic` branch snapshots inputs before the `short long` loop:

```bash
python3 -I "$D/bench/agentic_input_manifest.py" snapshot \
  --root "$D" --output "$O" > "$O/agentic_input_snapshot.log"
```

The existing replay reads the captured copy:

```bash
"$O/agentic_inputs/trajectories_${set}_production.json.xz"
```

After both replay sets, the runner verifies live inputs, captured copies, helper, and benchmark source. It writes verification atomically and propagates failures through `set -e` and cleanup:

```bash
python3 -I "$D/bench/agentic_input_manifest.py" verify \
  --root "$D" --output "$O" > "$O/agentic_input_verify.json.tmp"
mv "$O/agentic_input_verify.json.tmp" "$O/agentic_input_verify.json"
```

The runner retains per-set metrics before/after capture, the separate live-short smoke, source verification, and graceful FI shutdown verification. These steps are implemented; do not insert a second snapshot call. The helper refuses to overwrite an existing input snapshot. Do not attach a newly created manifest retroactively to an earlier replay: it cannot prove which bytes the earlier process consumed. Snapshotting before the replay and using the copies is the required integration contract.

The completed pair was compared after successful shutdown of both servers.
For a reproduction, choose a fresh output prefix; the recorded
`final_agentic_pair.{json,md}` already exists and is protected from overwrite:

```bash
python3 -I bench/compare_agentic.py BASE_RUN_ROOT CANDIDATE_RUN_ROOT \
  --output-prefix results/qwen_sequential/final_agentic_pair
python3 -I bench/compare_agentic.py --preflight
python3 -I bench/compare_agentic.py --preflight --production-inputs \
  > results/qwen_sequential/agentic_compare_cpu_preflight.json
```

The comparator writes `.json` and `.md` and refuses to overwrite existing report files. Both strict input identities and runtime source snapshots must pass. Controlled pairs additionally require matching fixed-reference FI mode/reference/runtime source hashes and successful startup/shutdown artifact verification; uncontrolled pairs must both be uncontrolled. Differences in observed Inductor selections are reported, not rejected merely because the graphs differ.

## Fixed workload

Each set has eight recorded trajectories. The replay queues `ceil(concurrency / 8)` copies of **all eight**, so concurrency 1 does not mean one trajectory. Every copied task must contain every recorded turn, even when the records arrive in a different thread completion order.

| set | concurrency | agents | turns | verified output tokens |
|---|---:|---:|---:|---:|
| short | 1 | 8 | 128 | 32,757 |
| short | 4 | 8 | 128 | 32,757 |
| short | 8 | 8 | 128 | 32,757 |
| short | 16 | 16 | 256 | 65,514 |
| short | 32 | 32 | 512 | 131,028 |
| long | 1 | 8 | 770 | 115,522 |
| long | 8 | 8 | 770 | 115,522 |
| long | 32 | 32 | 3,080 | 462,088 |

Compressed production input SHA256:

- Short: `e24df8aa498518add17b8188ef834ea4272ca78991513f6bbe0118f38021c680`
- Long: `3852ffe3133be6ad2c93442682906f45990944f86e4126ff10f9c7cbf83580d8`

The manifest also captures each turn's canonical token-ID hash, input/output lengths, task totals, the replay-only semantic hash, and helper/benchmark source bytes. Original `live_s` values and recorded tool-error statistics stay in the captured input bytes but are excluded from the replay semantic hash. They are prerecorded history, not new replay failures.

## Interpretation limits

The Markdown table reports all-turn TTFT p50/p90, turns/s, output tok/s, reported cache-hit rate, and wall time. The JSON also preserves first-turn mean TTFT and later-turn p50/p90. Percentiles use `sorted_values[int(n * p)]`, matching the benchmark. All-turn p90 is newly derived from detailed records.

Cache-hit fractions can be range-checked but not recomputed: the current report omits the per-concurrency raw prefix counters. MTP acceptance is `accepted tokens / drafted tokens`; acceptance length is `1 + accepted tokens / draft steps`. Runner metrics bracket an entire short or long set, so these are set-level aggregates, never per-concurrency MTP values.

The detailed replay records contain input lengths but do not echo token hashes, executed command lines, or cache salts. The required pre-run snapshot/post-run verification binds input bytes under the documented runner integration; it is not an independent server-side prompt receipt. For stronger future provenance, record the actual replay argv and generated cache namespace at execution, then include them in the report schema. The current PID-based cache salts are shared within each copied group as intended; fresh servers isolate the paired runs. This helper does not change salt behavior.

Replay tests fixed prompt/output-length performance. It neither checks generated-text equality nor establishes live agent quality. The existing `agentic_live_short.json` remains a separate functional smoke artifact; its tool results must be reviewed separately. A successful replay comparison does not turn that smoke into a quality-equivalence claim. One sequential pair also supplies no timing confidence interval.
