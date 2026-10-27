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
- Full individual pair is running under session61591: shared `b12x_tasks_1`
  finished17/17 serving scenarios plus20/20 needles; `next_tasks_1` is the live
  baseline. GPU budget0.78, explicit per-phase cache salts. No agentic stage.
- New scoped correctness follow-up, before the scalar experiment: original
  hybrid CUTLASS fallback computes `s2 / rounded_reciprocal(a)` instead of
  default `s2 * original_a`. CPU audit finds2335g1/2432g2 one-ULP differences.
  Isolated `shared_exact/` restores all20,480 alpha values exactly on CPU and
  leaves the b12x path unchanged; actual full modular-kernel GPU checks remain
  pending. Do not edit the running shared candidate or attribute throughput
  differences to this arithmetic without evidence.
- Installed vLLM0.32 bench does not default to greedy. Current repo protocol
  inherits checkpoint sampling temperature1.0/top_k20/top_p0.95; dataset seeds
  fix prompts. Preserve this common protocol for the running pair, and use
  explicit-greedy short decode diagnostics if the paired matrix still regresses.

- Full shared/next pair completed with both exit codes 0 and 17/17 scenarios.
  Quality and throughput comparison is being documented; no agentic run yet.
- Exact-alpha follow-up passed all 48 GPU comparisons on real layers 0/19,
  including full modular preparation/finalization and changed-route graph
  replay. Alphas exactly match default; most outputs have reduction noise.
  Storage alias checks confirm MMA scale views allocate no additional copies.
  `shared_exact` is now an explicit serving option; serving validation pending.
- Initial GDN numerical run started after exact-alpha exited, with tokens
  3/31/1025 and 8 warps. Scalar preparation is rebasing onto exact-alpha so its
  incremental comparison will not mix two changes.

- Scalar exact-alpha baseline: both GPU runs exited 0, 224 output comparisons
  and 40 dispatch checks passed. Full-call time improves about 3.2–3.4% at 4k,
  1.7–1.9% at 8k, and is neutral around ±0.5% at 16k. Incremental serving
  measurement is needed; no extrapolated server gain is claimed.
- GDN initial run exposed one FP8-byte mismatch under changed-input replay.
  Diagnosis found different RMSNorm reduction ordering before BF16 rounding;
  exact comparison remains required. Reproduction preserved, correction in
  CPU preparation. No GDN serving integration yet.
- A five-round cold NLL diagnostic is being prepared after quality probes
  exposed changes comparable with large within-backend variation. Next exact
  serving run will also include explicit temperature-zero decode diagnostics.
- Decode routing received the exclusive GPU slot after both scalar processes
  exited, for c1 control/target-only/full instrumentation only. No agentic
  replay/performance stage is scheduled at this point.

- Explicit-layout GDN correction now passes initial synthetic tests plus648
  real-weight synthetic cases and324 changed-input graph replays, including
  unchanged output GEMM bit checks. At layer18 with strided-gate copy included,
  warps8 full chain averages173.09→146.82us(4k),391.05→286.90us(8k),
  925.94→584.51us(16k). These are micro timings, not full-model improvements.
- Actual eager 8k activation capture v2 completed cleanly, followed by9 exact
  FP8/scale/GEMM comparisons and27 graph replays on layers0/18/38, all passing.
  First capture failed at RPC serialization of TorchVersion metadata; its
  evidence is preserved. Cast to str and bounded RPC timeouts fix that probe.
- Decode c1 pilot: control exited0, target-only exited1 because generated
  tokens differ from control. Raw64-step target routes are structurally valid,
  with zero graph padding in this sample.51/132 autotune tactics differ, so
  attribution is unresolved. Draft capture and control follow-up remain pending;
  no routing optimization is selected. See decode-route-capture/RESULTS.md.
- COMPLETED session28410 (exit0): exact_tasks_1(shared_exact individual+GREEDY_DECODE=1+
  NLL_ROUNDS=5), followed sequentially by next_diagnostic_1(next diagnostic
  with the same two diagnostic flags). Sources are snapshotted and verified
  at cleanup. Do not edit runner, serving matrix, source manifest helper,
  existing model overlays/build sources, or evaluation scripts until terminal.
  New isolated GDN overlay preparation is CPU-only during this queue.

- User explicitly confirmed compatible beneficial patches may be stacked.
  Measure incremental retained changes on the selected parent stack; final
  agentic remains one baseline-versus-composed-config stage. Do not assume
  isolated percentage gains add, or repeat agentic for each addition.
- Independent GDN integration review found actual next FX calls the opaque
  packed GDN norm core30times, bypassing _output_projection. Its prefill path
  directly invokes FLA despite norm.forward_native at construction. A3/B2
  traces each contain120 FLA layer_norm_fwd calls over4 requests. Thus the
  isolated oracle remains relevant, but integration must change the packed
  prefill boundary and preserve packed MTP decode. Initial consumer-hook draft
  remains unmounted and is being redesigned; no source in the active queue
  is modified. Evidence: gdn-norm-fp8/results/qa_interface_review_cpu.json.

- Exact full serving and greedy/NLL diagnostics completed; documentation and raw
  evidence checkpointed. Greedy decode differences are -0.73% to +1.41%; repeated
  cold NLL mean differences are smaller than observed within-backend ranges, not
  an equivalence confidence bound. Exact is the experimental stacking parent.
  Runtime audit finds43/132 FI tactic differences and4/13 Inductor selections;
  no shared/exact speed difference is attributed to alpha arithmetic.
- Compiled-output GDN fallback passed96 FP8/scale/real-GEMM comparisons across12
  shapes including36 changed-input graph replays. Generated kernels write Q/S
  directly; micro timings compare against separate copies, not native-only.
  Five-file overlay adopted it and passed9 CPU-only schema/dispatch checks.
- FAILED session54113 (exit1, GPU/container cleaned): gdn_integration_1, shared_exact_gdn diagnostic with explicit
  greedy decode and one cold NLL round. This is first stateful compiled integration,
  not a speed claim. Requires exactly30 ready target GDN layers before workloads.
  Runner/source helper/serve script and mounted overlays are frozen until terminal.
- A verified canonical FlashInfer table import for subsequent matched serving
  comparisons is in CPU preparation. Inductor settings remain observed, not pinned.

- First GDN integration activated30 layers; FX contains30 new packed ops with
  direct QA-to-CUTLASS consumers. All63 PIECEWISE captures succeeded. FULL
  MTP capture stopped on an unwarmed descriptor: MRV2 rounds184 to186 tokens
  for decode_query_len3. Warmup coverage fix is in CPU preparation; original
  failed run/source snapshots retained. No serving or speed claim for this run.

- FAILED session7614 (first startup exit1, no measured requests): scalar_matched_1 then exact_matched_1, each individual17
  serving tasks + legacyquality + explicitgreedydecode +NLL_ROUNDS3. FI_CONTROL1
  imports one immutable132-key reference table; startup and graceful-shutdown
  effective table gates must pass. Six importer files, runner/sourcehelper and
  mounted MoE/build sources are frozen. Inductor caches remain separate/unpinned.
- GDN CPU review also found explicit split config changes vLLM defaults: baseline
  adds two KV-update boundaries and disables rope-KV fusion. FirstGDN omitted
  them and enabled that unrelated fusion. Generator correction is in progress;
  retain baseline19 split ops and disabled rope fusion, then add only GDN op.

- FI startup check rejected20 changed effective tactics despite132/132 exact
  loaded entries. Dense FP4 uses cold-L2 tuning policy; FI intentionally ignores
  legacy persisted entries while tuning under that policy. Fixed-reference replay
  via native tune_mode=False and early loading is in CPU preparation; do not relax
  the table identity gates. Failed scalar_matched_1 evidence is checkpointed.
- COMPLETED session14108 (exit0): gdn_integration_2(shared_exact_gdn diagnostic, greedy1,NLL1,
  FI_CONTROL0). Warmup planner now covers81 sizes; completed baseline19 splits,
  rope-fusionFalse plus GDN op. This is integration validation, not a matched
  speed pair. Mounted GDN and runner/config sources frozen until terminal.

- Corrected GDN integration_2 completed all4 fixed-length greedy decode scenarios
  and10 cold NLL calls; all30 GDN layers enabled, source22/22 unchanged. This
  establishes compiled/stateful execution for these probes, not speed or model
  quality equivalence. Standalone numerical evidence remains separate.
- INCOMPLETE CONTROLLED PAIR session67599 (first run exit5): scalar_matched_2 followed by exact_matched_2, each full17
  tasks+legacyquality+greedy+NLL3. FI reference now preloads before model loading;
  native tune_modeFalse replays it during controlled warmup, preserving native
  tactic validation. Startup gate runs after final graph capture. Runtime six
  importer files, runner/sourcehelper, build/MoE overlays frozen until terminal.

- scalar_matched_2 completed17 tasks, legacyquality,4 greedydecode cases and30
  cold NLL calls; startup reference132/132 held aftercapture. The final export
  is missing: vLLM shutdown_timeout defaults0 and its API parent immediately
  SIGKILLed EngineCore afterSIGTERM, so Worker.shutdown never ran. Exit5 retains
  this limitation; exact_matched_2 was not launched. No matched pair result yet.
- Root adds supported --shutdown-timeout30 forFI_CONTROL1 (Docker grace45s).
  COMPLETED session2213 (exit0) fi_lifecycle_1 shared_exact diagnosticNLL1 is a short
  start/request/stop verification before another fullpair. Runtime importer6
  files andrunner/sourcehelper frozen during this smoke. No retroactive final
  tactic verification is manufactured for scalar_matched_2.
- Finalagentic comparator and immutabletrajectory snapshot helper are prepared
  CPU-only with37 negativefixtures/eight realworkloadshapes. Wiring is deferred
  untilafterindividual stack selection; noagenticGPUrun has occurred.

- FI lifecycle smoke passed both132-entry startup/shutdown gates and source
  verification; engine drained in30-second mode and exitedcleanly.
- Owned GDN packed-state probe1 failed before hookinstall: its four-mount
  baseline fixture incorrectly forced B12x for unquantized MTP experts. No
  comparison ran; ownedcontainer/GPUcleanup verified. Probe2 nowusesauto with
  a CPUmount/backend compatibility guard (9 tests); GPUretry awaits slot.
- COMPLETED ROOT GPU session43154 (both exit0): exact_matched_3 then scalar_matched_3, each
  fullindividual+greedy+NLL3, fixedFI replay+shutdown30. Allruntime sources
  frozen. Agentic input snapshot/verify was wired BEFORE this pair, but only
  the explicitly selected final agentic suite invokes it. Noagenticrun yet.

- exact_matched_3 completedexit0:17serving cases,4greedycases,30NLLcalls,
  source23/23 and both132-tactic gates pass. Strict comparator load on real
  artifacts passes. Session43154 now runs scalar_matched_3, samefrozenprotocol.

- scalar_matched_3 completedexit0:17tasks,4greedycases,30coldNLLcalls,
  source23/23, both132-tactic gates. Strict serving/greedy comparisons pass.
  Cold8k/16kTTFT -11.4%/-10.6%, but4/13Inductor choices differ and standalone
  microgain is much smaller. Both20/20needles; py_email30k NLL+0.007835 with
  nonoverlapping3-repeat ranges. Scalar is candidate, not quality-equivalence
  claim or default promotion. Agent maps changed compiler choices CPU-only.
- COMPLETED ROOT GPU session99954 (exit0): next_matched_3 next individual, FI1/greedy1/NLL3.
  Fresh controlled baseline for final composition. Runtime runner/sourcehelper,
  FI6/build/MoE/GDN sources frozen untilterminal. Noagenticrun.
- GDN probe2 CPU cleanup ownership/termination checks and strict fused-mixed
  coverage are being completed. Probe2 gets one GPU slot afterroot99954.

- CPU compiler audit corrects tuning-file interpretation for scalar_matched_3:
  13 wrappers per side linked; ten GDN artifacts have one identical candidate,
  three artifacts retain multiple identical candidate sets. Final in-memory
  choice remains unresolved for those three. Executable sections match;
  differing best_config files do not establish different runtime choices.
  Scalar chosen only as experimental parent for scalar_gdn incremental test;
  final adoption retains NLL flag and requires composed comparison to next.

- next_matched_3 completed all 17 tasks, four greedy cases, 30 cold NLL calls,
  20/20 needles, source21/21 and both132-tactic gates. Strict comparisons to
  exact/scalar pass. Scalar vs next cold8k/16k/32k -18.8%/-9.4%/-7.1%; workload
  throughput -1.4% to+5.2%, greedy rate -3.70% to+3.67%. Single configuration
  pair only; scalar py_email30k NLL+0.002717 vsnext overlaps observed ranges,
  while scalar-vs-exact flag remains. Logged nextKV44.61GiB vs45.50 for bothMoE.
- INCOMPLETE AGENT GPU session67555 (exit1, missing pure-long coverage): guarded private GDN packed_state_probe_2;
  exactly one granted attempt after root99954 cleanup/GPU vacancy. Runner uses
  immutable owned IDs for stop/removal, verified cleanup before lock release.
  No other GPU job. Next planned full serving candidate is scalar_gdn.

- GDN probe2 checked mixed_spec actual2048 (prefill2045+spec3) and puredecode6,
  each T/T+17: Q/S/private conv+SSM bits exact, padding invariant, live cache
  unchanged. It did not observe pure-long prefill, so coverage gate correctly
  fails. Owned container absent/compute empty; source copies match runtime hashes.
  Agent prepares a separate cold long-only request without weakening checks.
- COMPLETED AGENT GPU session23142 (exit0): offline fixed_plain_c1 control only, 192 tokens,
  original immutable route-control FI table and strict observed-lookup coverage.
  No aux/full automatic chain. GDN probe3 CPU-only preparation runs in parallel.

- fixed_plain_c1 completed 192 tokens and all six FI boundaries132/132,
  30,028 observed lookups over114 keys, zero misses/mismatches. Engine shutdown
  complete, owned container gone/compute empty. Aux/full still pending.
- COMPLETED AGENT GPU session52998 (exit0): private GDN packed_state_probe_3, unchanged
  numerical worker plus separate salted8192-token long-only request. One grant.
  Scalar/GDN reproducible patch packaging proceeds CPU-only without runtime edits.

- GDN probe3 passed mixed_spec2048, puredecode6, cold long_prefill2048 at
  T/T+17: all six Q/S/privateconv+SSM byte checks exact, padding/tails invariant,
  full live-cache pre/post identical. No checkpoint-bearing or mixed-non-spec
  metadata observed; scope layer18 eager/private state, not full model quality.
  Owned container absent/compute empty; GPU slot released.
- COMPLETED ROOT GPU session70620 (exit0): scalar_gdn_matched_3 scalar_gdn individual,
  FI1/greedy1/NLL3 and GPU budget0.78. All five GDN overlays, integration config,
  MoE/build sources, runner/sourcehelper/serve/FI6 frozen until terminal.
  Compare incrementally to scalar_matched_3 and overall to next_matched_3.
  Agentic still exclusively final stage; route aux/full follow after this run.

- Continuation revalidation: former tool session70620 is no longer addressable,
  but host runnerPID1616947 (sg parent1616945) and serving_matrixPID1623076 are
  confirmed live with exact scalar_gdn_matched_3 argv. Owned servercontainer is
  running and serving summary advances. Observe these processes/result markers;
  do not restart the experiment because the old tool handle disappeared.


### 2026-10-27 21:34 UTC: focused composition repeat

The full scalar/GDN composition completed with exit 0, all 17 serving cases, four greedy decode cases, 20/20 needles and 30 cold NLL calls. Incremental effects are mixed: 32k TTFT -7.6%, long-doc throughput +9.1%, but 8k TTFT +7.4% and c=64 greedy throughput -8.7%. GDN is not selected. A reversed-order focused repeat runs GDN first, scalar second, using all six unchanged prefill cases, four unchanged greedy cases and five cold NLL rounds. It omits the seven sampled throughput workloads and agentic. The default full17 comparator remains unchanged in scope; explicit `--suite prefill` validates only the six-case suite. CPU preflight and source hashes are recorded in `results/qwen_sequential/focused_repeat_cpu_preflight.json`.

Offline route controls fixed_aux_c1 and fixed_full_c1 have terminated with output-comparison failures; both owned shutdown APIs completed and the GPU is empty. Their FI/structural findings are being audited separately. No routing optimization is selected. Agentic remains the final next-versus-selected-composition pair only.

- COMPLETED ROOT GPU session28460 (both exit0): `gdn_focus_1` then `scalar_focus_1`, focused suite, FI1/NLL5. Runtime/source manifest files remain frozen through both runs. Final agentic CPU review finds no launch blocker; explicit FI1, GREEDY0, NLL0, NOSPEC0 for both final runs. Live-short smoke statistics require separate review even when process exit is zero.


### 2026-10-27 21:51 UTC: final composition selected

Both focused runs completed six prefill cases, four greedy cases, 50 cold NLL calls and source/FI startup+shutdown verification. Strict focused comparisons pass. GDN improves focused 8k/32k TTFT by7.31%/1.98%, but greedy output rates c4/c16/c64 regress4.18%/7.70%/4.04%. The c4/c64 penalty repeats the full pair; 8k prefill changes sign between pairs, while16k stays nearneutral. Five-round licenses30k incremental NLL delta is+.000125 with overlapping observed ranges; that does not retroactively erase the original stack-vs-next three-round flag or establish broad quality equivalence.

Select `scalar` as the final experimental composition: root next patches01-05 plus MoE03 scale/hybrid,04 sharedworkspace,05 exactCUTLASSalpha and06 scalarinputscale. Leave GDN default-off/unselected for this general serving stack. It remains a measured opt-in candidate with prefill wins and decode tradeoffs. Decode routing has no selected optimization.

Final authorized agentic pair is `next_final_agentic_1 next agentic`, then `stack_final_agentic_1 scalar agentic`, both FI_CONTROL1/GREEDY_DECODE0/NLL_ROUNDS0/NOSPEC0, memorybudget0.78 and existing fixed production trajectories. No per-patch agentic runs. Freeze all recorded runtime/harness/source bytes until both terminal; review live smoke separately from replay.

- LIVE ROOT GPU session11044: final pair launched sequentially with the explicit flags above; `next_final_agentic_1` is replaying short trajectories. The first c1 checkpoint completes128 turns, zero failed/unrun, metric_version2 valid. No final comparison until both runs and owned cleanup finish.
