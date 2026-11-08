# Native long-first task-order serving screen

Pair3 completed and remains unselected. Default OFF; no production selection. The full2048 diagnostic proves identical weighted BF16 contributions for the actual L19/M4096 capture across both schedules/repetitions, while final atomic sums differ by rounding order. This motivates a real serving screen, not a quality verdict.

Both controls mount the same plugin/source files and use the current scalar + CONFIGnext parent, official vLLM0.32 image, fixed FI reference, static checkpoint NVFP4 activation scale policy, hybrid threshold1024, and fixed MTP2. Only `TASK_ORDER_PROFILE=on` replaces the native gated constructor with the frozen uninstrumented long-first class. Native source hashes are checked against the original micro experiment. Routing, scale preparation, backend selection, GEMM math and atomic scatter are unchanged. No diagnostic scratch/stores are installed.

A shared explicit cache family keeps generated producers reusable across the controls; compile-factor SHA and FI compiler configuration audit must be compared rather than assumed equivalent. Maxbatch8192 permits one intact4096-token prefill. Fresh pair2 includes explicit4096 PIECEWISE capture and requires actual graph-node execution evidence; see the scope correction below.

`tasks.py` uses the repository real-text prompt, exact4096 token inputs and one output token. Two warmups precede eight measured serial requests. Prefix cache is explicitly reset before each request. Timing is **client nonstream request latency**, not streaming TTFT. Four cold NLL repetitions use identical token IDs and must be compared against within-control variability, not merely a single A/B logit difference. A separate final4096 request is traced and excluded from latency results.

Evidence is fail-closed: startup requires the intended native class to be instantiated; NSYS requires actual gated kernel launches of the expected class during the live request; explicit post-workload RPC reads the existing native workspace after synchronization, verifies routed-row count4096×8 and all published packed descriptors against the CPU scheduler, and requires a nonidentity split/unsplit queue. The observer adds only a Python reference assignment during wrapper acquisition; GPU readback occurs after measurement. Scratch shared across layers retains the last native call, so queue readback proves that retained actual call, not every layer independently. Named native trace launches provide separate execution evidence.

One control per exclusive grant:

```bash
CACHE_FAMILY=task-order-pair1 sg docker -c 'bash kernels/moe-task-order-serving/run.sh task_order_serving_a1 off'
```

Root schedules OFF/ON/ON/OFF using unique result tags and the same cache family; this runner never autochains. Source snapshots, image identity, exact owned cleanup and strict FI shutdown verification remain required. The explicit `server_post_workload` snapshot is separate and never relabeled as shutdown. Agentic remains a final selected-composition task, not part of this screen.

CPU checks: Python AST/compile and shell syntax; both plugin profiles register with CUDA hidden and no CUDA initialization. GPU launch and artifact interpretation still require root review/grant.

## First OFF attempt and shutdown correction

`task_order_serving_a1` completed all requests, live trace and explicit post-workload verification but exited5 because the strict worker shutdown snapshot was absent. Sources verified and owned cleanup left GPU/container empty. The live trace contains 40 actual native gated kernels totaling17.292 ms; the retained4096 queue has T405/S330/735 tasks and matches baseline descriptors exactly. Median cold nonstream request latency was94.482 ms over eight requests; this is a control measurement, not a candidate gain.

Baseline identical-token NLL itself varied across four cold repeats:0.240264,0.219173,0.199898,0.191320. Differences are distributed across positions; the mechanism is unresolved. Do not attribute this to the scheduler candidate, or infer state leakage merely from the NLL trend. The fresh baseline will retain the same four-repeat numerical protocol.

The shutdown log ordering is consistent with nsys PID1 forwarding SIGTERM to the process tree while the API process also signals EngineCore, which can receive the second signal after restoring its default handler. The harness now discovers the unique Python/vLLM process owning the container's TCP LISTEN8000 socket after health-ready, records PID/start ticks/executable/cmdline hash/socket inode, then revalidates and sends SIGTERM through a pidfd to that API process only. It waits up to60 seconds for natural container exit; any failure/timeout is recorded and falls back to stopping/removing only the owned container. Strict shutdown verification is unchanged. CPU tests on host and inside the existing container verify exact signaling and stale-identity rejection. This correction has not yet been exercised on a live serving run; use a new cache family/source pair.

### Fresh pair2 graph coverage correction

The original a1 ladder ended at128, so its4096 prefill target ran eagerly: root trace analysis found all40 native MoE launches without graphNodeId. That run is activation/latency diagnostic evidence, not representative graph-prefill performance. Pair2 adds explicit4096 capture while retaining the numerical/workload protocol and the API-only shutdown correction. This is a source/cache pair reset.

Installed next-runner prefill uses `ModelCudaGraphManager.run_pw_graph` (PIECEWISE), not the FULL decode replay method. The plugin records the actual worker's4096 PIECEWISE capture plan but explicitly does not call a configured plan proof of execution. A minimal live NVTX marker around that existing replay boundary carries the actual input tensor length. The trace gate requires every native target launch to have graphNodeId>0 and to correlate through an actual `cudaGraphLaunch` inside the live4096 marker, with at least40 backbone native launches. If this fails, the run is ineligible for graph-prefill attribution; it must not silently become an eager performance claim.

## Pair2 outcomes and cache-identity correction

`task_order_serving_a2` executed the real4096 graph path: all40 native target nodes were captured and correlated to cudaGraphLaunch inside the live4096 PIECEWISE marker. Median cold request latency76.419 ms; NLL repeats0.208465/0.183689/0.228015/0.188414. Its original exit1 is preserved: the offline checker used `r.name` instead of the actual NSYS `r.nameId` column. Corrected CPU revalidation passed and is saved separately in `trace_check_revalidated.json`; post-workload queue RPC was skipped after that checker error. API-only graceful exit succeeded naturally and strict FI shutdown132 passed. This is partial-control evidence, not a relabeled successful terminal.

`task_order_serving_b2` is **invalid as ON evidence**: startup constructed LongFirst, but live trace correctly rejected40 baseline native kernel symbols. The two cache identities were incomplete. The profile was absent from vLLM compile factors, and installed FlashInfer's native dynamic disk key encodes geometry/scales/activation but not the implementation source; its compiler helper can load an existing baseline object after constructing the candidate object. A plain-byte AOT scan found no native symbols, so the exact AOT embedding contribution is not established. No serving gain or numerical effect can be assigned to long-first from b2. Strict shutdown/cleanup passed; the activation failure remains exit1.

The same generated `envs_task_order.py` is now mounted in both profiles and registers default-off `VLLM_SM120_TASK_ORDER_LONG_FIRST`, asserted consistent with the declared profile. The native dynamic key is extended by the exact selected source SHA; its original geometry/algorithm fields remain intact. Other static/micro native keys are unchanged. Registration requires the dynamic memory cache to be empty. CPU tests against the actual installed key builder show distinct OFF/ON compile-factor hashes and native disk filenames; see `cache_identity_cpu.json`. Actual execution must still pass the existing graph/class and published-queue guards. This correction requires a fresh matched source/cache family; no corrected GPU retry has yet run.

## Pair3 complete screen — unselected

The corrected matched family `task-order-pair3` ran ON b3 followed by OFF a3. Both exited0, with identical harness/source/image/checkpoint identities, exact prompt IDs, strict FI132 shutdown, natural API-only shutdown, and empty owned container/GPU after cleanup. The intended compile-factor and native source keys differ. Live4096 traces prove40 captured target nodes in each control; the final retained queues match the selected descriptor order. This validates activation, not production selection.

| Metric | OFF a3 | ON b3 |
|---|---:|---:|
| Median cold nonstream request latency, eight requests | 69.000 ms | 76.754 ms |
| Sum of40 target native kernel durations, separate trace | 17.226 ms | 17.026 ms |
| Four cold NLL repeats | .235198/.252023/.209555/.183405 | .229887/.197617/.220878/.198475 |

The target-duration sum decreased1.16%, while client latency increased11.24%. These are separate observations, not a causal attribution: among25 standalone generated Triton functions per profile,10 had mapped best-config artifacts and exact common source identity; two unchanged producers selected different configurations. The pointwise selection changed from XBLOCK1024/4 warps OFF to512/8 ON, and a reduction changed R0_BLOCK1024/8 warps to4096/16. The exact comparison is archived under `results/task_order_serving_b3/inductor_config_comparison_complete.json`; the paired summary is `results/task_order_serving_pair3_summary.json`.

No repeat is scheduled and long-first remains unselected. This one ON/OFF startup pair does not establish a serving gain, quality equivalence, or regression caused specifically by task order. NLL varies within each control, and summed target durations are not critical-path time. Earlier failed/partial runs retain their original statuses. Agentic remains reserved for a selected final composition.
