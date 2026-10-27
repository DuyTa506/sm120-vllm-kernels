# Qwen prefill workspace implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Share b12x scratch safely across Qwen MoE layers and measure the candidate in vLLM.

**Architecture:** Keep per-layer wrappers/weight caches, backed by shared scratch
owned by the current vLLM WorkspaceManager execution slot. Resolve ownership
at apply time; preserve the small-batch CUTLASS path and copy results out.

**Tech Stack:** Python, PyTorch, FlashInfer B12xMoEWrapper, vLLM v0.32.0, SM120.

**Spec:** `docs/superpowers/specs/2026-10-27-qwen-prefill-workspace.md`

## Global constraints

- Target `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast`, one SM120 GPU, TP=1/EP=1.
- Preserve routing, weights, scales, MTP-2 and small-batch CUTLASS dispatch.
- Opt-in flag `VLLM_B12X_SHARED_WORKSPACE=1`; baseline stays available.
- Depend on verified shared-buffer and WorkspaceManager persistent-resource APIs.
- GPU tests wait for unrelated training to exit; no production server changes.

## Review focus

- Same shape but different layer: scratch may alias, weight caches must not.
- Different owner/configuration: scratch and temporary output must not alias.
- Decode-only calls: b12x scratch must not be allocated unnecessarily.
- Locked/reset manager: no stale owner reuse or silent allocation after lock.
- Graph replay/returned outputs: no stale routing or overwrite by a later layer.

### Task 1: Allocation regression and backend fix

Files: candidate `kernels/moe-nvfp4/shared/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py`,
`kernels/moe-nvfp4/test_shared_workspace.py`, incremental patch 04.

- [x] Write GPU regression using the real wrapper: the second identical layer
  must add <16 MiB; legacy allocations should violate this assertion.
- [x] Run it against the unchanged candidate and record the expected failure.
- [x] Implement owner-scoped buffer and per-layer wrapper caches; move lazy
  construction below the small-batch return.
- [x] Test same-owner reuse, shape/lane/ubatch isolation, manager lock/reset,
  default-off behavior and decode bypass.
- [x] Generate incremental patch against the existing scale-corrected backend;
  verify clean application and byte equality with the candidate overlay.

### Task 2: Kernel and server validation

Files: `kernels/moe-nvfp4/verify_shared_workspace.py`, focused A/B runner,
`kernels/moe-nvfp4/SHARED_WORKSPACE.md`, raw results.

- [x] Run real Qwen layer pairs through unchanged/shared wrappers; compare
  eager and graph outputs across routing changes and hybrid boundaries.
- [x] Measure persistent allocation and interleaved MoE timings.
- [ ] Run CONFIG=next baseline/candidate with isolated compile caches;
  record KV tokens, 8k/16k TTFT, GPU span and decode regression checks.
- [x] Run independent code review of the complete diff and resolve substantive
  findings with regression coverage.
- [ ] Document exact commands, results and remaining limitations; keep candidate
  opt-in until full serving evidence supports enabling it.

## Execution log

- User said “cứ làm đi” and prioritized prefill; proceed without additional
  design/execution-method questions. Worktree: `.build/worktrees/qwen-prefill`,
  branch `perf/qwen-prefill-workspace`; main checkout and serving are untouched.
- User subsequently authorized shared-GPU measurements; report load/noise and use two timing orders. Never stop the unrelated workload.
- No centralized test suite exists in this benchmark repository. Run the
  focused allocation/kernel suite and existing relevant numerical checks.

- Regression evidence: decode allocation test failed before the fix; GPU suite had 5 expected failures and 2 passes before sharing. After the fix, all 8 tests pass. Second-layer allocation: 579.652 MiB → 0.000 MiB.
- Real Qwen layers 0/19: 80 routing/replay comparisons passed at 1023/1024/1025/8192 tokens. Baseline repeated-run noise is recorded separately. Two-layer scratch: 1159.305 → 579.652 MiB. CONFIG=next server A/B is running.

- Independent review: no critical backend defects. Runtime resource lookup is inside the opaque fused-MoE custom op. Corrected benchmark readiness/warmup reuse of input_requests[0] by disabling both in measured invocations and warming separate seeds. Added detailed samples and GPU telemetry. Initial A1 is excluded from speedup conclusions (cached first prompt; runner aborted after an in-flight script edit). Future runner files remain unchanged until exit.
- Review ruling: strengthen the output-lifetime check to call a candidate layer with a fresh output, then rerun numerical validation. Concurrent streams sharing one owner, EP/general output dtypes, mutable model configurations and sleep-mode reclamation remain unsupported/unvalidated outside the pinned target. Absolute model accuracy is inherited from the existing scale-corrected backend; these tests verify consistency, not higher-precision model quality.
- Extended validation after the output-lifetime fix: 30 additional routing/replay comparisons at 3/12/16384 tokens passed; total 110 comparisons.
- A2 and B1 finished successfully. Under concurrent training, the candidate's two 8k mean TTFTs are 9.06%/8.85% lower, with similar full-server KV capacity. This is exploratory, not a clean speedup claim. Both 16k runs reuse seeds from 8k and can hit prefix cache. The attempted reset endpoint returned 404 and did not reset anything. Cold-v2 uses unique seeds per length plus decode warmup; bash syntax passes. Waiting for training completion and GPU priority before further GPU runs.
