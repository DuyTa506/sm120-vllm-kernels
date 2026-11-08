# Independent ON3 evidence review (CPU only)

Reviewed `dp4a-expert-reuse/serving/results/fi_direct_serving_on3` and current post-workload verifier/launcher without editing the active serving subtree or using GPU.

## What is supported

ON3 remains **exit5**, with missing legacy shutdown checkpoint. Its three benchmark tasks completed their exact requested counts and lengths, and the trace contains4,960 native-direct events plus4,960 sanitizer events. Independently querying SQLite correlates **all4,960 sanitizer events**, as well as the existing native count, to the live `FI_DIRECT_LIVE_GRAPH_M3` marker through runtime correlation and graphNodeId>0. This supports actual M3 graph integration, not merely setup/JIT execution.

The current `verify_post_workload.py` passes offline on preserved ON3. The explicit RPC result equals the worker-written file; startup/post worker PID396, resources, compile-factor digest and four runtime runner/workspace/speculator source hashes agree. There are120 prepared resources (40 owners/layers×M1/M3/M4 for the recorded owner scope), and warmup input/output pointers are released. Startup and post FI effective entries both match the fixed132-entry reference digest. Therefore the measurement-period state is verified even though shutdown-hook delivery is missing. Do not relabel the RPC as shutdown, rewrite exit5, or claim a complete legacy lifecycle.

Current launcher changes compared with ON3 snapshot are confined to the post-workload verifier and cleanup accounting: the old shutdown check is retained as a separate diagnostic; strict post-workload verification now controls the measurement-period exit criterion. Existing files changed since ON3 are README and run.sh only; candidate generated runtime code is unchanged. These changes occur after the workloads and do not themselves alter their measured work. OFF1 should still archive the changed harness rather than claim byte-identical launcher manifests.

## Active compiler-path check

ON3 docker.log saves backbone AOT `1caa76d5fe69ea6805605d0f0ed884470c1fe23b010103a19fb1700e01afb260` and draft AOT `8bed63ae8e7f80bac80bb9c456e16d7235a17f521fb1d3c90dc86a14a4a1e633`, using backbone/eagle cache directory35d71dba45. These exactly match the two IDs in `active_inductor_configs.json`. All13 selected-config records are inside those active AOT directories. The cumulative cache manifest also contains older paths and must not be treated as execution evidence.

This is a sound active-directory filter. It does not prove every recorded best_config was the actual first-launch binary configuration: no tuner-state→wrapper→PTX/cubin→launch binding or producer pin is claimed. OFF comparison must identify unchanged common producer bodies and compare their selected configs, not simply count matching filenames or compare cumulative directories. If common producers differ, report the serving delta as confounded; FI tactic equality does not control Inductor tuning. Runtime env flag should differ, candidate sources should match, and distinct compile-factor/cache identities should be verified.

## Limits of a speed claim

- This is one short sequential ON screen: decode_c1 about3.27s, decode_c4 about2.73s, prefill about0.88s. A single OFF pair can show a provisional direction but does not establish repeatability or isolate small deltas from drift.
- The profiling request is separate and follows timed workloads. It proves live M3 activation for that C1 trace. Post-workload counters also show M18/M9/M6; those shapes fall outside the M1/M3/M4 candidate. Do not infer all C4 kernels took the candidate or that all prepared shapes were exercised live. All40 setup resources are not, alone, per-layer runtime activation proof.
- The workloads are random-token generation with fixed output lengths, not text quality/NLL or agentic tasks. Matching completed counts and lengths is required but does not establish equivalent generated content or speculative acceptance behavior. Quality and any accepted-token changes need separate measurement before promotion.
- Prefix caching is enabled. Installed `vllm/benchmarks/serve.py:843-917` creates its initial test input from `input_requests[0]`, then reuses that same input for all8 warmups. The runner has no prefix-cache reset between tasks. Thus `prefill_2048` is warmup-enabled with prefix caching enabled; actual cache hits were not independently quantified. The parent hybrid cache block is2176 while this prompt is2048, so repeated prompts alone do not prove cache reuse. Treat this as a focused fallback nonregression check, not a clean cold-prefill result. Do not change this protocol mid-pair; disclose it, then use a separately designed cold/warm task later if needed.
- The current graph ladder stops at32, unlike the full production ladder. Compare this focused serving configuration faithfully, not the unrestricted production configuration. Larger prefill is outside the direct candidate geometry.
- The post verifier robustly checks fixed measurement-boundary state, but it is not a continuous assertion over every dispatch. Source files, image, source controls, trace and actual output checks provide complementary evidence.

## OFF1 comparison checklist

Match model checkpoint/image, same generated candidate files with flagoff, parent scalar source controls, MTP2 runner/speculator, graph ladder, maxseq8/batched16384/maxlen8192, prompt seeds and lengths, concurrency/task order/warmups/prefix-cache policy, FI132 fixed reference and both post-boundary checks. OFF trace must contain no native-direct or sanitizer candidate events while retaining live graph markers. Preserve exact cleanup/source verification and the historical ON3 exit5. Compare active common Inductor producers after OFF finishes; label any difference explicitly. Report TPOT/TTFT/throughput with their raw task scope and avoid treating the microbenchmark ratio as expected whole-model speedup.

No source changes or extra GPU runs were needed for this review.
