# Isolated opt-in serving integration (not yet GPU/serving validated)

`prepare.py` composes local copies of the skinny round2 generated vLLM files with a new DG helper. It records base file hashes and exports the patch; neither installed vLLM nor the other agent's source tree is edited.

Enable `VLLM_SM120_DG_ROWWISE_FP8=1`. This registered environment flag changes `vllm.envs.compile_factors()`; CPU verification observed False→True. The helper's custom-op schema and flag-off import passed CPU-only checks. Both sources and compile factors must be identical across paired controls except the explicitly tested flag.

Supported model scope: QKVZ B=[2048,12288], column-major FP8 E4M3, contiguous dynamic per-token FP32 activation scales, contiguous FP32 per-channel weight scales, BF16 output, no bias, SM120. Initial M set is {32,33,48,63,64,65,80,96,112,128}; unknown shapes fall through. The underlying isolated C++ binding rejects split-K/AB swap/accumulation.

`process_weights_after_loading` invokes preparation once per GPU: unit blockscale metadata for every accepted M and shared B scale metadata are allocated and retained, and the fused kernels are warmed before CUDA graph construction. There is no FP32 output scratch. Output allocation follows the existing returned-tensor custom-op contract and uses graph memory pools under capture.

Dispatch order for eligible QKVZ layers is DG on supported cells, then the existing skinny custom op only when that independently enabled layer guard passed, then CUTLASS. Bias and non-BF16 output bypass DG entirely. Flag-off keeps the composed skinny source behavior. This does not assert that skinny is active in any pre-existing serving configuration.

Mount generated `vllm/envs.py`, `.../cutlass.py`, `.../sm120_dg_rowwise_fp8.py`, and the paired skinny `.py/.cu` from `generated/` into an owned serving container, together with the isolated `fused_src` at `/work/src` and `PYTHONPATH=/work/src`. Use an isolated DG cache and control image/toolchain. Root owns this launch and GPU slot; no serving launch has occurred.

`dispatch_audit()` reports Python dispatch and graph-construction calls only. It does not count graph replay executions. Live GPU traces must show `EpilogueRowChannel` on the intended request shapes before claiming an active serving optimization. M48/80/112 were added only after their actual-operand bucket checks passed; intended serving shapes still need live GPU-path proof. Actual serving/NLL checks, flag-off equivalence, warm/cold graph behavior and representative task micro-effects remain pending. Agentic remains the final composed evaluation.

## Controlled initial serving pair

`run.sh TAG off|on` owns one server and requires an explicit `CACHE_FAMILY`. Both profiles mount the same generated source and isolated extension, use `dg-sm120-env:1`, scalar+CONFIGnext, fixed FlashInfer import table and fixed MTP2; skinny flags stay off for this first pair. It snapshots integration/base/source/extension evidence, verifies after run, and owns cleanup under the shared GPU lock.

Each run performs a focused prefill2048 and decode-c16 micro (shorter decode outputs than the full serving matrix), plus cold real-text prompt NLL at48/64/96/128 tokens so the eligible GEMM is actually exercised. This is an initial focused screen, not the full repository suite. Live NSYS node tracing includes NVTX around the actual next-runner `ModelCudaGraphManager.run_fullgraph` after warmup. The gate correlates captured `EpilogueRowChannel` GPU kernels to runtime launches inside observed M48/64 replay ranges; unused labels or eager-only execution do not pass. Flag-off must execute no candidate kernel. CPU tests cover those false-positive cases.

Example after a root GPU grant, from the worktree:

```bash
CACHE_FAMILY=dg-r2-pair1 sg docker -c 'bash kernels/deepgemm-sm120/integration/run.sh dg_serving_off1 off'
```

Run `on` only after a separate grant and baseline cleanup. No agentic evaluation is included.

## Scope and full-suite expansion after the initial pair

The focused prefill2048 case is **fallback nonregression**: 2048 is outside the accepted DG M set. It cannot establish direct candidate prefill acceleration. Small real-text NLL lengths48/64/96/128 exercise the candidate during prefill, but are numerical probes rather than sustained prefill-throughput claims. The initial short decode-c16 measurement is also an eligibility/performance screen, not a complete serving result.

If the focused pair passes activation and quality gates, the next controlled pair should restore the repository production-sized max context/capture ladder and run `bench/serving_matrix.py --suite all` with its existing target-length validation, distinct warmup/measured seeds and per-phase cache salts. It covers seven throughput workloads (including RAG/shared-prefix/chat/mixed lengths), six prefill lengths2048/4096/5000/8192/16384/32768 and decode c1/c4/c16/c64. Preserve both warmup and measured records; the shared-prefix workload tests intentional reuse, while phase salts prevent accidental warmup contamination. Independently repeat controls in reversed order after startup; compare observed DG-active shape frequencies, fallback coverage and GPU state before attributing changes.

Quality expansion should run `eval/quality_probe.py` (12 real text/code/math/Vietnamese prompts), `eval/long_eval.py` (long-document NLL plus needle retrieval) and `eval/repeated_nll.py` for repeated cold prompt NLL where differences approach noise. Keep the eligible-length NLL probe because long prompts mostly hit fallback and cannot alone validate the modified GEMM. Inspect current-model generated outputs and NLL alongside baseline self-repeat variation. Do not use random-token throughput or the microbench speedup alone as model-quality evidence.

Task-level comparisons come before stacking/promoting the patch. Combined agentic evaluation remains last after the selected patches are composed and their individual effects established.

CPU verification is invoked directly (the trace script uses top-level assertions, not unittest discovery):

```bash
python3 kernels/deepgemm-sm120/integration/test_trace_gate.py
bash -n kernels/deepgemm-sm120/integration/run.sh
```

### First serving screen and preparation correction

`dg_serving_off1` completed the short focused screen and live trace gate: 30 QKVZ modules matched, candidate/skinny disabled, live M48 full graph replay observed, no fused candidate kernels. Scalar source verification passed (23 files), fixed FI shutdown verification passed (132 entries), and cleanup left no owned container/GPU process. The short baseline measured prefill mean TTFT 105.239 ms and decode 1810.24 output tokens/s, TPOT 6.7865 ms; these are screening observations, not a speedup claim.

`dg_serving_on1` exited 1 during weight preparation before serving: vLLM set default dtype to BF16, and two unit-scale `torch.ones` calls inherited it. Native scale packing requires float32. Failed logs and source snapshots remain immutable. Both unit-scale allocations now explicitly specify float32; activations, output and row scales already had explicit dtypes. `test_preparation_dtype.py` executes the real preparation function with CPU doubles only at CUDA/native boundaries under default BF16 and checks every allocation's explicit dtype. It passed without CUDA initialization.

The startup logs also exposed a policy difference: standalone DeepJIT defaults PDL false, while vLLM's DeepGEMM lazy initialization enables it on SM120. The matched experiment plugin now installs `fixed_pdl_policy.py` before model load, forcing false for both existing and later initialization. Preparation and startup/shutdown audits assert the effective policy. CPU `test_fixed_pdl.py` validates late initialization and idempotence in both profiles. The policy is not changed inside any graph/dispatch path. Because sources and global policy changed, the next comparison must be a fresh matched pair with a new explicit cache family (`dg-r2-pair2`); off1 is retained only as earlier evidence.

### Corrected serving pair: candidate executes, serving effect inconclusive

`dg_serving_off2`/`dg_serving_on2` used the same image, byte-identical experiment manifests and isolated DeepGEMM source fingerprints, fixed PDL false, fixed FI startup reference and cache family `dg-r2-pair2`. All 30 matched QKVZ modules enabled on the candidate. Both traces contain 13 live M48 full graph replays. ON has exactly 30 named fused candidate nodes per replay. Median candidate summed duration is 0.584697 ms versus 0.665625 ms for OFF's inferred QKVZ geometry (12.16% lower, about 80.9 µs over 30 calls). OFF identification is geometry-based, and summed kernel durations do not establish critical-path savings.

| Focused task | OFF | ON | Interpretation |
|---|---:|---:|---|
| Prefill2048 mean TTFT | 190.335 ms | 190.862 ms | Fallback nonregression screen; +0.28% |
| Decode c16 output throughput | 1378.75 tok/s | 1388.17 tok/s | +0.68%, one short sample |
| Decode c16 mean TPOT | 8.2102 ms | 8.5738 ms | +4.43%, worse |

Identical-token NLL changes ON−OFF at 48/64/96/128 tokens were +0.004059/+0.000709/+0.009164/+0.035570; the largest token logprob difference was 4.921. This does not establish acceptable model-quality equivalence. It also cannot be attributed solely to DeepGEMM: among 10 exactly mapped common-source Inductor selected configs, three differ. In particular, the unchanged gate+FP8 reduction body selected R0_BLOCK4096/16 warps on OFF and R0_BLOCK1024/8 warps on ON. The comparison enumerates 25 standalone sources per profile; it does not claim complete runtime kernel coverage.

OFF exited0 with shutdown provenance. ON completed requests/NLL/trace but exited5 because both worker-level shutdown snapshots were missing, despite no Python exception and successful cleanup/source verification. Its lifecycle remains incompletely verified. A plausible race exists when EngineCore resets SIGTERM to the default before resource teardown and the API manager sends another SIGTERM; this is an inference from source and log ordering, not a proven root cause.

The candidate remains **default-off and unselected**. Structured evidence is `results/dg_serving_pair2_summary.json`, ON's `fused_geometry_profile.json`, `inductor_config_comparison_complete.json`, and the preserved source/cache snapshots. No agentic test was run.

A CPU-prepared follow-up uses the existing loopback-only `/collective_rpc` route after all measured tasks to invoke a fixed worker audit method. It writes a distinct `server_post_workload` FI checkpoint plus DG runtime evidence while the worker is alive, explicitly reports `shutdown_proven=false`, and retains the strict separate shutdown verifier. `test_post_workload.py` passes missing-checkpoint/FI-drift rejection using temporary synthetic copies; no live follow-up has run. Optional static mapping to the existing PDL pin reference covers only 2 of 16 observed standalone kernel identities; missing identities must fail rather than fall back silently. New recording/pinned serving runs are deferred pending higher-value candidates.
