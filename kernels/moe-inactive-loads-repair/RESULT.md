# First GPU pair: resource repair succeeds, no full-call gain

`defined_fc2_off_1` and `defined_fc2_on_1` both exited0, cleaned their exact owned process groups and left GPU empty. Frozen14-file manifests,13 runtime/helper SHA maps, checkpoint and capture identity match. Actual CUDA graph traces identify baseline and `MoEGatedDynamicKernelDefinedFC2` respectively at both M1024 and M4096. Candidate SHA remains `db422c6ea3c32a58e9a93339ec66d369955ddf6b43a69b41815821bb75c8eb1e`.

| L19 tokens, tile128 | OFF graph span µs, three block medians | ON graph span µs, three block medians | Median-of-block change |
|---|---|---|---:|
| 1024 | 314.192 / 313.776 / 313.872 | 315.024 / 314.736 / 314.832 | +0.306% |
| 4096 | 437.248 / 437.504 / 436.976 | 442.640 / 442.880 / 443.152 | +1.288% |

There is no positive full-call signal. M1024 is effectively neutral at this single-pair resolution; M4096 trends slower. These are one sequential OFF/ON fresh-process pair, not repeated startup significance estimates. The candidate is not selected for serving or agentic composition. No further GPU run was started.

All8 input-hash-matched output comparisons passed the exploratory finite/relL2<=.02 gate, maximum paired relL2=.00437766. Original, modified input/routes/weights, padded-tail routes and restored scenarios were exercised. This does not establish production quality equivalence or before-atomic contribution correctness. Those more expensive diagnostics are deferred because this variant did not win.

## Actual loaded artifact evidence

The host-side artifact inspector extracted complete embedded CUDA executables from each persisted host object, hashed both levels and successfully disassembled them. Actual graph trace names match the inspected class symbols. Resources match the conclusion of the matched-fast_mathTrue offline pair:

| Static evidence | OFF | ON |
|---|---:|---:|
| REG | 168 | 168 |
| STACK bytes | 80 | 80 |
| Static SHARED bytes | 1024 | 1024 |
| LDL / STL instruction sites | 4 / 5 | 4 / 5 |
| LDSM / OMMA instruction sites | 384 / 896 | 384 / 896 |
| BRA instruction sites | 628 | 644 |
| All instruction sites | 12057 | 12281 |

The catastrophic local-memory growth of the original31-guard variant is absent. Narrowing the patch and defining zero values removed that resource regression in this combined change; the experiment does not isolate which change was responsible. Extra branching/zero-definition work remains, and fewer dynamically useful operand loads did not produce a measured full-call gain. Static counts are not dynamic execution/stall measurements and cannot independently explain the remaining latency difference.

`results/paired_gpu_1.json` contains numerical/timing comparisons. Each run's `compiled_artifact_0` contains resource output, full SASS, extracted cubin and SHA binding. `results/gpu_publication_summary.json` binds these plus source/runtime manifests, original outputs and actual profiler traces for review. The failed initial experiment remains untouched in `../moe-inactive-loads`.
