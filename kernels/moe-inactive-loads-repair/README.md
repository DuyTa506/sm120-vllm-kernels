# Narrow FC2 defined-register experiment

CPU-only, compiled for SM120a with CUDA hidden. No GPU numerical result, timing, serving integration or promotion.

The failed31-guard experiment in `../moe-inactive-loads` remains closed and unchanged. This new full-class candidate changes only six FC2-tail B/SFB shared-to-register copy sites. Each destination register slice is first reinterpreted as Uint32 and filled with zero bits, then overwritten by the original copy when `valid_rows > warp_m_coord*16`. All FC1 paths and common FC2 A/SFA loading are baseline code. The M128 constructor guard remains; K2048/I512 is explicitly checked. Class name is `MoEGatedDynamicKernelDefinedFC2`, source SHA `db422c6ea3c32a58e9a93339ec66d369955ddf6b43a69b41815821bb75c8eb1e`.

The zero definition seeks to avoid carrying irrelevant previous fragment contents through conditional SSA joins. Active warps immediately overwrite zeros before MMA; fully inactive warps consume no fragment. Destinations are zeroed at the same register slice that the original copy writes, immediately before that copy, with no intervening use. FP4/FP8 zero bits represent zero; no numeric conversion to FP4 is requested. The synthetic predicate truth table covers every valid_rows1..128 and all8 MMA warps. This is source/dataflow evidence, not an executed numerical proof.

Five tests verify deterministic generation, inverse-AST equality to the entire baseline, unchanged waits, correct zero/copy destinations and fragment-consumption conditions. They reject deliberate destination and synchronization mutations. No scheduling, producer, fence, pipeline-state, accumulator, quantization or output-store statement is otherwise modified.

## Offline compilation evidence

`compile_offline.py` copies the installed dispatcher's fake-tensor compile function into an isolated namespace. Only its host device queries are replaced with previously recorded188-SM metadata; fake tensors, launch arguments and options come from that function. `CUDA_VISIBLE_DEVICES=''`, explicit `CUTE_DSL_ARCH=sm_120a`, and a rejecting Torch lazy-init function prevent enabling CUDA. Reports record `torch_cuda_initialized=false`. It compiles/exports but never invokes the kernel or shared disk-cache loader. Candidate factory replacement exists only in this disposable CPU process.

The final comparison uses serving's actual `fast_math=True` default for both modes, reciprocal scales, shared input, E256/top8/K2048/I512/M128, fakeM1024/maxrows131072, opt-level2 and TVM-FFI. Native wrapper does not override fast_math; installed launch and compile methods defaultTrue. Baseline and candidate use identical fake metadata and options. Offline cubins need not be byte-identical to prior runtime cubins; these resource claims are specifically the matched offline pair.

| Compiled static evidence | Baseline True | Repair True |
|---|---:|---:|
| REG | 168 | 168 |
| STACK bytes | 80 | 80 |
| Static SHARED bytes | 1024 | 1024 |
| LDL/STL sites | 4 / 5 | 4 / 5 |
| LDSM / OMMA sites | 384 / 896 | 384 / 896 |
| BRA sites | 628 | 644 |
| All instruction sites | 12065 | 12289 |
| Cubin bytes | 292704 | 298632 |

Complete SASS/resources and extraction/object/cubin hashes are in `results/baseline_fast_inspection` and `results/repair_fast_inspection`. Counts are static sites, not dynamic execution counts. Static shared allocation excludes dynamic shared memory. Unchanged LDL/STL/stack removes the catastrophic resource regression seen previously, but extra branches/zero definitions can still lose. There is no speed claim. Narrowing scope and zero-definition were both changed, so this comparison does not isolate which prevented spill growth.

Preserved failures: `offline_compile_1` rejects Float4 `.fill(0)` directly; reconstructed exact failed source snapshot is retained beside it with its checked SHA. `offline_compile_2` compiled but export omitted a required function name. `offline_compile_3` exported the corrected bitwise-zero version with fast_mathFalse. Matched `offline_baseline_1` shows False itself produces STACK144 and LDL15/STL16 for both modes, so comparing that repair to prior servingTrue baseline would be invalid. The final True pair resolves that confound.

An isolated micrograph launcher is now prepared; no GPU job or serving plugin has run. A root-reviewed later experiment must verify actual candidate graph activation, fixed source-qualified caches, numerical input/route changes and complete expert-call latency before considering any integration. A positive latency result would still require numerical self-noise and before-atomic contribution diagnostics.

## Prepared isolated GPU screen (requires root grant)

Run exactly one mode per granted slot:

```bash
sg docker -c 'bash kernels/moe-inactive-loads-repair/run.sh defined_fc2_off_1 off'
sg docker -c 'bash kernels/moe-inactive-loads-repair/run.sh defined_fc2_on_1 on'
```

The copied harness retains full scalar calls, L19/M1024/M4096, unchanged M128, four scenarios each, fresh-process separate cache roots, original fixed scales, profiler-after-timing and exact owned-process cleanup. Source-qualified dynamic keys append `defined_fc2_source_sha256` in both modes; default is OFF. Native factory class is replaced directly only in ON; no task hooks. Actual profiler events must include DefinedFC2 in ON and exclude it in OFF.

Compared with the failed experiment harness, only candidate paths/class/key, a strict runtime-hash parity check in the comparator, and derived artifact inspection have changed. After the GPU child exits, host `artifacts.py` extracts complete embedded cubins and requires successful resource/SASS inspection using host CUDA tools. This postprocessing executes no GPU work. Both input sources and all derived artifact hashes are retained; the original failed directory is untouched.

`compare.py OFF_DIR ON_DIR` now automatically checks full runtime SHA maps, capture identity and source snapshot parity, in addition to identical8 input scenarios and exploratory output gates. It ignores only CPU unittest elapsed-time metadata, not executable source differences. Positive speed would still require review/repetition and before-atomic numerical diagnostics; unchanged compile resources are not performance proof.
