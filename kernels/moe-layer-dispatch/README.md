# Per-layer NVFP4 prefill backend calibration

Status: GPU screen completed: CUTLASS was slower in all12 sampled cells, so no dispatch policy was promoted. Repeatability diagnostics also completed; same-input variation occurs in eager and graph modes with both shared and private workspace. Details and preserved failure records follow below. No runtime patch is installed.

Hypothesis: some layer/token shapes run faster through the existing CUTLASS fallback than through scalar b12x, even above the global 1024-token threshold. Calibrate a frozen exact-shape table using actual checkpoint weights and captured MoE inputs. Retain the current path below 1024 and scalar b12x for unlisted shapes. Exact shapes avoid unsupported extrapolation between token buckets.

`calibrate.py` calls the existing scalar backend's complete `apply` and `_apply_cutlass`, including activation quantization, dynamic setup, GEMMs and finalization. Weight preparation, routing and graph construction are outside timing for both. This isolates the experts call, not a complete routed/shared-expert layer or vLLM request. Source hashes and checkpoint/capture provenance are saved.

Initial subset: layers 0/19/39, M=1024/2048; then 4096 and other captured layers if the effect survives. The existing container capture contains layers 0/9/19/29/39, each 8192x2048 (CPU inspected, no CUDA initialization). Calibration uses rows in the first half and validation uses rows in the second half, without token repetition. These are distinct real activations/routing but the same prompt; independent-prompt generalization remains a later gate. Supply `--heldout-capture` for a separate capture; duplicate file hashes are rejected.

The policy is frozen on disk before held-out timings. CUTLASS must be at least 5% faster in both alternating benchmark orders and pass the relative-L2 accuracy gate to be selected. Held-out timing cannot change selection. Autotuning runs only on calibration inputs. Held-out correctness must pass; isolated cross-backend numerical agreement does not replace vLLM task-quality checks. Candidate selection is not a promotion recommendation until the held-out paired orders also show a stable benefit. `measurements.jsonl` retains unsuccessful cells.

CPU verification:

```bash
python3 -m unittest discover -s kernels/moe-layer-dispatch -v
python3 -m py_compile kernels/moe-layer-dispatch/calibrate.py
```

GPU command for the root scheduler (do not run concurrently with another experiment):

```bash
sg docker -c 'docker exec qwen-prefill-dev python3 /work/kernels/moe-layer-dispatch/calibrate.py --capture /work/kernels/moe-nvfp4/data/moe_inputs.pt --layers 0 19 39 --tokens 1024 2048 --repeats 30 --replays 5 --output /work/kernels/moe-layer-dispatch/results/initial'
```

The persistent container already sees a readonly 166MiB capture at `/work/kernels/moe-nvfp4/data/moe_inputs.pt`; the host worktree placeholder is zero bytes. A fresh container must mount the main checkout's nonempty capture there. Do not overwrite the placeholder or dataset. Output directory must be new so previous trials cannot be silently replaced.

Integration guard, after evidence: bind a stable layer prefix once in `process_weights_after_loading(layer)`; load/validate a frozen table before graph compilation; include `cache_identity(policy)` and checkpoint/backend provenance in compile-cache configuration; dispatch only on host-known layer and exact `hidden_states.shape[0]`. Verify prefix mapping against actual model layers. Never inspect GPU expert counts on the host during forward. `policy.select` rejects a changed backend source, retains the existing below1024 branch and defaults unknown cells to scalar. The scalar input scale eligibility checks must remain intact. Source-hash protection alone is insufficient for production: model revision, TP/EP geometry, runtime build and policy hash must also match.

Before merging: replay fixed graphs with changed same-shape real routing, compare the full routed/shared-expert layer (including auxiliary-stream overlap), run repository vLLM prefill/decode micro tasks with a fresh compile cache, then stack passing patches. Agentic remains the final combined-stack measurement.

Root review guards: this is a hot-cache isolated-layer screen, not evidence of a cold full-model gain. Each changed-input graph is also checked against its own backend fresh eager execution, preserving graph outputs before scratch reuse. Helper/harness hashes are recorded; any numerical failure writes the summary then exits nonzero. CPU-inspected persistent-container capture is nonempty; verify that mount before launching a new container.

Root-scheduled locked launcher (preferred over the bare command above):

```bash
sg docker -c 'bash kernels/moe-layer-dispatch/run_calibration.sh layer_dispatch_1'
```

It holds the exact shared GPU lock, checks vacancy, validates the real capture on CPU, archives executable sources, records container/GPU state and owns only the calibration subprocess group inside the persistent container. PID start ticks and process-group identity guard cleanup; the dev container is not removed. Terminal code, process record and source verification are retained under the unique tag. CPU preflight passed with CUDA uninitialized and capture SHA256 `0e161c36a6f478c0205b3a7e8b388296eccafaa2d5999cf4420fcbcd0aa2f4ac`.

## Initial measured screen: rejected for sampled cells

`results/layer_dispatch_1` completed all12 cells and exited1 on its strict numerical gate; source verification passed and the GPU compute list was empty after cleanup. No policy entry selected CUTLASS. Both timing orders show CUTLASS slower for every sampled layer/shape on calibration and held-out activations. This is a hot-cache screen and does not exhaust other shapes or routing distributions.

| phase | layer | M | scalar us (orders AB/BA) | CUTLASS us (orders AB/BA) |
|---|---:|---:|---:|---:|
| calibration | 0 | 1024 | 337.1/337.3 | 354.5/354.4 |
| calibration | 0 | 2048 | 399.3/398.8 | 413.5/413.3 |
| calibration | 19 | 1024 | 314.8/315.1 | 322.7/320.9 |
| calibration | 19 | 2048 | 339.8/339.4 | 388.4/388.5 |
| calibration | 39 | 1024 | 317.5/317.1 | 339.8/339.5 |
| calibration | 39 | 2048 | 377.8/378.0 | 395.2/396.3 |
| heldout | 0 | 1024 | 340.3/339.4 | 362.1/362.2 |
| heldout | 0 | 2048 | 401.3/400.6 | 418.5/418.7 |
| heldout | 19 | 1024 | 313.1/314.4 | 329.9/329.9 |
| heldout | 19 | 2048 | 374.4/373.8 | 392.2/390.7 |
| heldout | 39 | 1024 | 316.6/315.8 | 339.5/339.0 |
| heldout | 39 | 2048 | 380.6/380.5 | 403.0/403.0 |

Numerical diagnosis is separate from this performance rejection. The cross-backend residual around2% is consistent with the existing `moe-nvfp4/README.md:226-247` study, which reports1.6–2.7% from different FP4 rounding boundaries. The initial0.002 cross-backend gate was too strict to characterize that already-known contract; it has not been relaxed and no policy is promoted. This observation does not establish a new kernel bug.

Same-backend graph/eager comparison was also nonexact on both backends. Installed scalar gated kernel uses BF16 atomic scatter (`blackwell_sm12x/_moe_dynamic/gated.py:163-197`, weighted scatter near2655/2697), and CUTLASS's optional fused-finalize epilogue selects BF16 RED_ADD operations (`epilogue_moe_finalize.hpp:406-429`). Non-associative accumulation order is therefore a concrete explanation to test, not a proven diagnosis of these particular outputs: the exact selected CUTLASS tactic was not traced in the initial screen, and shared scratch or tactic differences must be excluded.

`diagnostic/repeat_oracle.py` is CPU-prepared, GPU-unrun. Smallest next diagnostic: L39/M1024, eight eager repeats, eight graph repeats and graph/eager pairs, each in a fresh process with `--workspace shared` then `--workspace private`; records selected autotuner runner/tactic/input shapes/cache extras during eager and graph capture. Outputs are cloned before any scratch reuse. If same-mode repeats already vary comparably and private buffers do not remove it, that supports accumulation-order noise; if only graph/eager differs, inspect tactic/path selection. This diagnostic never changes the policy gate and its successful exit only means finite diagnostic data, not bitwise correctness. Requires a new root GPU grant.

### Repeatability diagnostic resolved the graph-specific concern

`layer_repeat_oracle_1` failed in observer code on an optional `None` input; its exit1 and logs are retained. The observer was fixed and CPU-tested to preserve optional/scalar positions without modifying selection. `layer_repeat_oracle_2` then completed both fresh processes, exit0, source verification passed and GPU-after empty.

L39/M1024 median relative-L2 repeat differences:

| backend / workspace | eager–eager | graph–graph | graph–eager |
|---|---:|---:|---:|
| scalar / shared | 0.003383 | 0.003440 | 0.003460 |
| scalar / private | 0.003391 | 0.003422 | 0.003330 |
| CUTLASS / shared | 0.000531 | 0.000972 | 0.000558 |
| CUTLASS / private | 0.000549 | 0.000839 | 0.000605 |

All repeated comparisons were finite and non-bitwise-identical, including eager–eager. CUTLASS selected GEMM1 tactic18 and GEMM2 tactic57 consistently across eager/capture and both workspace modes. Private allocation does not eliminate the variation. Together with atomic-finalize source evidence, this supports ordinary accumulation-order variability rather than a graph-specific stale-input or shared-workspace-only failure. This is a supported diagnosis, not an atomic-path ablation proof. It does not explain away the separate ~2% cross-backend FP4 contract difference. Initial policy accuracy gates remain unchanged and all sampled policy cells remain rejected on performance regardless.
