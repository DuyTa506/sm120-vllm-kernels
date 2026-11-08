Latest pressure attempt `replay_r2_preempt1` exited1 because observed preemption delta remained0. It passed numerical checks over908,610 state rows and390 intermediate-boundary rows; all32 boundary requests produced2304 tokens, stress91 completed/37 aborted with zero failures, and post-stress probe passed. Actual startup geometry matched the strict50-block/16384-context guard. The scheduler admitted about4 requests and queued the remainder instead of evicting them. Cleanup/source verification passed and GPU was empty. This result does not satisfy the preemption gate; timing remains blocked on actual eviction coverage.

Current validated result: `replay_r2_shadow3` completed with exit0, 869,070 real checked state rows and 630 intermediate-boundary rows; state/output/armed/boundary errors all zero. All32 boundary requests produced2304 tokens; stress completed91 and aborted37 with zero failures, followed by the post-stress probe. Runtime guard confirmed actual V2 runner, MTP2, all30 FP32 GDN state layers and composed source hashes. Source verification passed; owned container was removed and GPU process list empty. This is the shadow/state correctness gate only: preemption delta was0 and serving timing has not been measured.

`replay_r2_shadow2` separately passed full startup graph capture and runtime guard but exited1 before boundary requests: newly added random-range-ratio1 was invalid for installed vLLM0.32. Ratio0 produces fixed lengths. `check_sampler_cpu.py` executes installed sampler bodies and confirms32 exact128-input/2304-output requests; ratio1 rejection is retained as negative evidence. No actual request-boundary correctness evidence came from shadow2. All11 guard tests passed before shadow3.

Current GPU status: `replay_r2_shadow` exited 1 during startup graph capture (37/63), with a 384 MiB shadow snapshot allocation failing. Startup logged 2,040 compared rows with zero mismatches, **zero intermediate-boundary rows**; this is incomplete validation, not a correctness pass. Source snapshots and cleanup evidence remain under that result tag.

CPU revision after failure scopes diagnostic suspension to `Worker.compile_or_warm_up_model`: dummy forwards execute authoritative stock without retaining shadow snapshots, because capture does not perform normal acceptance/commit. The flag restores in `finally`; preexisting or unexpected pending snapshots fail instead of being cleared. All 64 real-request rows, intermediate-boundary equality, abort/stress workload and final gates remain required. Native replay library/scratch still prepare through layer registration; startup guard explicitly verifies the library exists. This shadow-only change provides no timing claim, and a new GPU run needs a fresh grant. Seven CPU guard tests pass.

# Current-scalar + next GDN replay composition

Status: CPU preparation only; no current-stack GPU result. Historical replay results remain unchanged. Fixed MTP2 only; dynamic MTP is not composed.

`compose.py` verifies exact audit hashes for next GDN, pristine envs/model-state and original replay patch, then applies with fuzz0 into `generated/`. The isolated serve script mounts composed next GDN plus original next attention/conv helpers. Runtime sources outside round2 are unchanged. The replay flag remains registered in envs for compilation keys; each mode/tag also uses a fresh compilation-cache namespace.

The shadow-only extension checks up to64 rows rather than the historical first8, compares final accepted state and intermediate checkpoint state using the same documented token-position formula, and raises on state/marker/output mismatches. Boundary counts are emitted explicitly. Stock remains authoritative in CHECK2. Numerical serving paths and CUDA replay source are unchanged. The all-row shadow can require substantial temporary memory; it is diagnostic, never a timed candidate.

Prepared launch sequence (root must grant each GPU run):

```bash
sg docker -c 'bash kernels/gdn-replay/round2/run.sh replay_r2_shadow shadow'
# Choose KVBLOCKS after observing available blocks; do not guess a cap that prevents startup.
KVBLOCKS=<reviewed-cap> sg docker -c 'bash kernels/gdn-replay/round2/run.sh replay_r2_preempt preempt'
```

Both modes execute concurrent2304-token continuations to cross the2176 state boundary, mixed finishing/abort stress and post-stress probe. `check_gate.py` requires nonzero checked rows, no state/output/marker errors and actual checkpoint coverage in shadow mode. The separate preemption run MUST record a positive `num_preemptions_total` delta; successful stress without preemption does not pass. The historical stress script does not exit nonzero on failed requests; the round2 launcher parses its terminal summary and explicitly rejects missing summaries or failed requests. A measured preemption event plus clean shadow is still not proof that every resumed row was checked; request-level resume/state attribution remains a review requirement before promotion.

`off`/`on` serving mode refuses to start without explicit `GDN_SHADOW_GATE` and `GDN_PREEMPT_GATE` JSON paths whose modes/pass statuses match. The gates carry the composition hash and timing mode requires matching composition plus completed exit0 records from both preceding runs. A startup-only general plugin verifies the actual baseline V2 runner, fixed MTP2 without dynamic schedule, PP1, CUDA GDN path and fp32 registered state layers; runtime class/source hashes are written for review. Once authorized, c48/c64 retain current scalar+next/FI controls, full graphs and fixed MTP2, with default replay MIN_REQS32. No agentic work is scheduled here.

CPU composed-env compile-factor comparison passed (distinct flag0/1 hashes, CUDA uninitialized); exact evidence and isolated-module limitation are in `compile_key_cpu.json`. Runtime guard, source-bound gate attestation and fail-closed stress assertions are implemented but GPU-unrun. Pending before execution: review debug boundary checker and select an explicit preemption cap after actual cache capacity is known. No claim that source applicability proves state correctness or speed.

Readiness update: every textual composition replacement now requires exactly one match. A fresh guarded composition under `/tmp/gdn-round2-guarded-compose-20261028` reproduced every generated byte and the original composition manifest exactly. Startup hashes the actual installed files against every non-backup `vllm/` entry in that manifest, including replay CUDA/Python, GDN, envs, model-state, attention and conv helpers; matching the environment SHA alone is insufficient. Enabled mode requires exactly30 FP32 GDN layers. CPU negative tests reject changed runtime bytes, wrong manifest identity, missing/duplicate replacement sites and29-layer coverage.

Next shadow command, awaiting root grant:

```bash
sg docker -c 'bash kernels/gdn-replay/round2/run.sh replay_r2_shadow shadow'
```

Estimated reservation:10–20minutes including fresh compile/JIT and diagnostic shadow overhead; this is not a measured duration. The boundary workload starts32 requests at c16, each128 prompt tokens plus2304 forced decode tokens, so each sequence crosses the2176-token checkpoint while multiple rows remain eligible for replay. MTP2 permits accepted spans of1–3 tokens; some crossings must have an intermediate boundary (`0<m<nu`) for the gate to pass. Merely crossing exactly at a final accepted slot is insufficient; if the count stays zero, retain failure and design a targeted offset sweep rather than assert coverage. Pure decode steady portions allow the replay path; prefill/mixed steps keep existing fallback. Additional abort/finish stress and post-stress probe follow. This shadow run does not claim preemption coverage: that is a separate capped-cache run before any c48/c64 timing.
