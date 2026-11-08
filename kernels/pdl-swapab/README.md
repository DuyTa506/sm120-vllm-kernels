# pdl-swapab: two DeepGEMM ideas (PDL, swap-AB) for MTP-2 decode on SM120

Config: `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast`, vLLM v0.32.0 image, RTX PRO 6000 Blackwell (SM120, 188 SMs),
MTP-2, CUDA graphs. The original experiments below reused PDL/swap-AB ideas rather
than integrating DeepGEMM code. The current DeepGEMM `nv_dev` branch includes
SM120 kernels; that implementation is evaluated in the separate DeepGEMM track.
Historical measurements below are retained with their original scope; the round-2
sections distinguish completed GPU diagnostics from CPU-only preparation.

## Round 2 status (2026-10-28)

Actual vLLM PDL0/1 testing is now implemented in `run_vllm_pdl.sh`, with
explicit container PDL environment, fresh per-run compiler caches, frozen
launcher/helper snapshots and the fixed 132-entry FI reference. The first
PDL0/PDL1 diagnostics finished with exit 0; no serving speedup is established yet. See
[ROUND2_PLAN.md](ROUND2_PLAN.md) for the subsequent focused prefill pair.

`audit_pdl_artifacts.py` binds generated wrappers to the exact AOT model files
named by the current startup log, using pickle opcode inspection without
executing pickle code. PDL0 has 14 backbone and 2 eagle wrappers and no
PDL launch markers. This is saved-artifact evidence, not runtime overlap.
`compare_pdl.py` retains the common strict identity/workload/FI checks and
reports its narrow normalization of PDL/cache/container naming differences.

The historical small-M timings below used unit activation scales and a real
fourth token. They do **not** establish drop-in equivalence with this
checkpoint's calibrated scales, or that current c1 routes a padded fourth row.
Those contracts require separate tests before a serving dispatch is valid.

PDL0 baseline `pdl0_probe_1`: all four greedy cases and ten cold NLL requests
completed. Common strict identity/workload validation, 23 runtime-source hashes,
frozen PDL driver/helper hashes, and 132-entry FI startup/shutdown checks passed.
Owned container removed and compute GPU empty at lease release. NLL collection
is diagnostic, not a standalone quality pass.

| Greedy case | Output tokens/s | Median TPOT (ms) | MTP acceptance |
|---|---:|---:|---:|
| decode-c1 | 438.59 | 2.123 | 93.52% |
| decode-c4 | 1255.28 | 2.770 | 89.31% |
| decode-c16 | 2578.73 | 5.606 | 86.57% |
| decode-c64 | 4968.02 | 12.068 | 86.63% |

Evidence: [strict baseline validation](results/pdl0_probe_1/pdl_baseline_validation.json),
[AOT binding](results/pdl0_probe_1/pdl_aot_binding.json). Baseline only; await PDL1
and repeat evidence before deciding whether to keep the switch.

### First actual-vLLM pair: diagnostic, not a keep decision

Both runs completed with source/FI checks passing. PDL1 binds all 14 backbone
and 2 eagle AOT wrappers with wait/launch metadata; generated PTX contains
`griddepcontrol.wait` and `griddepcontrol.launch_dependents`. PDL0 has zero
participating wrappers. Actual overlapping consumer execution remains unproven.

| Case | PDL0 output tok/s | PDL1 output tok/s | Change | Identical generated texts |
|---|---:|---:|---:|---:|
| c1 | 438.59 | 442.52 | +0.90% | 2/8 |
| c4 | 1255.28 | 1270.21 | +1.19% | 6/24 |
| c16 | 2578.73 | 2561.12 | -0.68% | 16/96 |
| c64 | 4968.02 | 5180.90 | +4.29% | 60/384 |

The ten cold NLL prompts were measured once per mode. `logprobs-2` rises from
0.358821 to 0.432513 (+0.073692); `logprobs-0` rises by 0.037769. No same-mode
noise range exists for this pair, so these differences cannot be dismissed and
no quality equivalence is established. Repeated same-mode cold NLL is required
before interpreting the mode difference.

FI tactics were fixed, but Inductor was **not pinned**. All 13 observed tuning
entries map into 8 matching function-body groups after removing only GDC wait/
signal expression statements; 3 groups select different configurations:
zeroing 1024/4-warps → 512/8-warps, one reduction selection 2/16-warps →
1/8-warps, and mul/silu/slice XBLOCK 256 → 128. This is observed `best_config`
evidence, not an exhaustive binary/runtime-selection comparison. The timing
and NLL differences cannot be attributed solely to PDL.

Evidence: [strict pair](results/pdl_probe_pair_1.json),
[NLL](results/pdl_probe_pair_1.nll.json),
[codegen/config audit](results/pdl_probe_pair_1.codegen.json).
A separate frozen `run_vllm_pdl_focused.sh` is ready for the original six-case
prefill sweep plus four greedy cases; queue the reverse order PDL1 then PDL0.
The first diagnostic had no prefill timing sweep. Agentic remains final composed
stack only, and PDL is not promoted by this result.

### Controlled Inductor pair (NLL/config smoke completed; no timing claim)

`pdl_record0_1` stopped during compile with a dtype-serialization error; its
failure artifacts are retained. The fix explicitly encodes Torch/Triton dtype
constants and rejects unknown object types. A CPU regression reconstructs the
exact failed QK/rope constructor without initializing CUDA.

`pdl_record0_2` completed startup plus ten cold NLL requests with exit 0,
source/FI/event gates passing. It records 18 full kernel identities; all have
constructor and first-launch evidence. The deterministic intersection reference
is [pdl_controlled_reference_v1.json](results/pdl_controlled_reference_v1.json),
SHA256 `6cfd53ede3b3b99ce7a858ab2c78e067100ce20c697188ebd1da4e9bc317ac27`.
This deliberately defines a **new controlled baseline**, not a replay of the
historical winning configurations. Empty intersections and unknown identities
fail closed.

`pdl_pin0_1` and `pdl_pin1_1` completed with exit 0, the same six frozen
executable sources/reference, 23 runtime-source checks, and fixed 132-entry FI
startup/shutdown verification. Both have 18 constructor and first-launch
identities with identical selected configurations. The strict
[pair report](results/pdl_pin_pair_1.json) passes its configuration contract.

All 16 actual saved backbone/eagle wrappers map to the same 12 source identities.
PDL0 additionally serializes 96 singleton autotuner states; PDL1 instead stores
compiled Triton bundles without those states. The audit therefore checks both
formats explicitly: 107 PDL0 / 111 PDL1 bundled records bind exact generated
source locations and full constructor identities to bundled PTX/cubin hashes,
compiled warps/CTAs/stages, and guarded first-launch singleton configurations.
Shared compiled fragments are matched across the same saved model. Full constexpr
choices come from actual first-launch events, not inferred from warp count.
The two extra wrapper identities were absent from the older serialized-state-only
count. Missing/changed source, wrapper digest, bundle or singleton choices fail
closed; nine CPU negative/positive fixtures cover these boundaries.

Singleton compilation creates no `.best_config` files. The directory-local
comparator replaces only the common nonempty-file assumption with complete
AOT/reference/first-launch evidence. FI/source/runtime checks remain intact and
original artifacts are preserved.

**Numerical behavior remains unresolved.** Exact prompt descriptors, probe
source, request settings and scored token-ID hashes match for all ten prompts.
For example, `logprobs-2` changes **0.428639 → 0.282102** (−0.146537);
`logprobs-3` changes 0.572588 → 0.584938 (+0.012350). This is not tokenization
drift, and a lower NLL is not proof of equivalent computation. Each side has one
cold round, so neither self-noise nor layer reproducibility is established.
[NLL details](results/pdl_pin_pair_1.nll.json).

No prefill/greedy throughput was measured in this pinned smoke. The separate
`run_vllm_pdl_pin_focused.sh` prepares six prefill cases, four greedy cases and
**five cold NLL rounds** per mode. Unknown lazy identities abort: if new workload
identities appear, record the extended workload and establish a new common
reference before rerunning both controls. No agentic benchmark is run here.

### Native swap-AB ablation (two fresh-process matrices completed)

[SWAP_AB_ROUND2.md](SWAP_AB_ROUND2.md) separates native same-format FP8/FP4
eligibility from the rowwise adapter and calibrated NVFP4 model contracts.
The isolated native override extension SHA256 is
`93488bde73201c38ffc670ff79ee14089052f5aa7a9fd0a396b603b232a3137c`.
Forced off/on keeps the original eligibility guard, identical FP8 bytes/scales
within each comparison, PDL disabled and a weight pool of at least 640 MiB.
Both matrices use 25 alternating off/on samples per case; the second fresh
process reverses shape and M order. Sources, binary and all 6,946 fingerprinted
entries match. Both exit/cleanup codes are 0 and GPU vacancy is recorded.

| M | N×K | off µs, run 1 / 2 | on µs, run 1 / 2 | latency reduction, run 1 / 2 |
|---:|---:|---:|---:|---:|
| 1 | 2048×4096 | 10.870 / 8.579 | 8.351 / 8.312 | 23.17% / 3.11% |
| 3 | 2048×4096 | 9.459 / 9.334 | 8.410 / 8.388 | 11.09% / 10.13% |
| 8 | 2048×4096 | 9.325 / 9.444 | 8.386 / 8.377 | 10.08% / 11.30% |
| 16 | 2048×4096 | 10.266 / 8.715 | 8.550 / 8.574 | 16.71% / 1.62% |
| 1 | 12288×2048 | 18.098 / 18.084 | 17.522 / 17.536 | 3.18% / 3.03% |
| 3 | 12288×2048 | 18.394 / 18.111 | 17.646 / 17.596 | 4.07% / 2.84% |
| 8 | 12288×2048 | 18.036 / 18.136 | 17.556 / 17.594 | 2.66% / 2.99% |
| 16 | 12288×2048 | 18.177 / 18.135 | 17.711 / 17.711 | 2.56% / 2.33% |

All sixteen comparisons have **bit-identical off/on BF16 outputs** and pass
changed-input CUDA-graph replay. Against FP32 dequantized operands, relative RMS
error is at most 0.001705. This establishes this native UE8M0 FP8 mechanism,
not numerical equivalence for calibrated NVFP4 checkpoint inference.

The selected policies differ beyond orientation. N2048×K4096 changes
64×64×128 / 5 stages to effective weight-major 64×16×64 / 16 stages;
N12288×K2048 changes 64×128×128 / 3 stages to 128×16×64 / 10 stages.
Generated template/cubin audit shows **split-K 4 for both orientations at
N2048×K4096, and split-K 1 for both at N12288×K2048**. The original
`DG_PRINT_CONFIGS` trace printed the default split-K before finalization; it is
preserved as observed and is not authoritative for this field. Exact per-cell
source/cubin hashes and final template parameters are in the repeated report.
These are **total swap-policy effects**. The initial one-cell
smoke showed 22.85% at M3, but the matrix repeats show about 10–11% there;
M1/M16 non-swap timings also vary substantially by process/order. Report the
paired run values, not a pooled maximum. Fresh reversed-cell RNG consumption
also differs, so operands are identical off/on within each case, not asserted
identical across processes.

Evidence: [validated repeated matrix](results/swap_native_pair_1.json),
[first matrix](results/swap_native_matrix_1/results.json),
[reverse matrix](results/swap_native_matrix_2/results.json),
[initial smoke](results/swap_native_smoke_1/results.json).
Reproduce with `run_swap_ab_control.sh TAG --ms 1,3,8,16 --shapes
2048x4096,12288x2048 --iterations 25`, then a new tag and reversed cell order.

**Native auto-swap already selects this policy.** This experiment validates its
benefit; it does not create a new vLLM patch. Porting it to the same-byte rowwise
adapter still requires exchanging row/column scale axes and checking transposed
stores; native UE8M0 scales cannot substitute for checkpoint NVFP4 scales.

### Same-byte rowwise swap prototype (first actual-operand smoke completed)

The native ablation led to a separate model-format-compatible candidate:
[rowwise_swap_control/PLAN.md](rowwise_swap_control/PLAN.md). A new M1..16 API
retains actual FP8 bytes and FP32 token/channel scales. A dedicated swapped
epilogue preserves token-then-channel multiplication order while the original
M>=32 API remains unchanged. Split-K and accumulation remain rejected.

The isolated extension build and five NVCC device-template instantiations pass
without a GPU; four CPU fixtures and actual capture/comparator provenance checks
also pass. The replay targets real QKVZ layer0/9/29 slices against CUTLASS and the exact
skinny `(1,4)` dispatch, with changed activation/scales and cold CUDA graphs.

The first smoke, **layer0/M3**, completed exit0/cleanup0 with all 20 frozen inputs
and source fingerprint verified. DG off/on outputs are bit-identical to CUTLASS
for the original input, a second actual activation slice, zero token scale and
nonuniform token/channel scales. Every changed-input graph matches eager; M0/17
reject. Full FP32-reference relative-L2 is at most 0.001686 on these cases.

| backend | forward order µs | reverse order µs |
|---|---:|---:|
| CUTLASS | 20.041 | 20.094 |
| existing skinny `(1,4)` | **17.001** | **17.039** |
| DG rowwise off | 18.481 | 18.472 |
| DG rowwise swap | 17.765 | 17.805 |

Swap improves the DG operator by **3.61–3.87%** and beats CUTLASS by about 11.4%,
but is **4.50% slower than the existing skinny candidate**. Thus this smoke does
not justify replacing the composed dispatch. It establishes a functioning
same-byte small-M rowwise swap contract; other M/layers are not measured yet.
[Raw result](results/rowwise_swap_smoke_1/results.json),
[validated comparison](results/rowwise_swap_smoke_1/analysis.json).
No vLLM integration has been performed. The focused PDL runner remains frozen
and queued separately.

The separate [rowwise_tile_control](rowwise_tile_control/PLAN.md) candidate now
has an actual layer0/M3 GPU smoke. Its valid64×16×64 swapped policy raises logical
work tiles from96 to192 within the same188-CTA persistent grid. Final trace and
generated template binding confirm88320B shared memory/16stages/split-K1; AUTO
uses98816B/10stages/split-K1. All original/changed-input/scale checks pass, with
DG outputs bit-identical to CUTLASS.

| backend | forward order µs | reverse order µs |
|---|---:|---:|
| CUTLASS | 19.877 | 19.941 |
| existing skinny | **16.904** | **16.891** |
| DG rowwise off | 18.346 | 18.337 |
| DG AUTO swap | 17.640 | 17.624 |
| DG forced64×16 swap | 19.620 | 19.622 |

The narrower tile is **11.22–11.33% slower than AUTO** and **16.07–16.17% slower
than skinny**. Increasing first-wave active work did not improve this operator.
This candidate is not promoted; its complete result remains recorded.
[Raw smoke](results/rowwise_tile_smoke_1/results.json),
[validated comparison and finalized policies](results/rowwise_tile_smoke_1/analysis.json).
Exit/cleanup0, frozen20-source checks and source fingerprint pass; owned container
removed and GPU empty at lease release. No vLLM micro-effect is claimed.

## Historical summary


| idea | verdict | size of the prize (c=1 step, ~6.4 ms) | what it would take |
|---|---|---|---|
| **A. PDL** | The idle gap inside the decode graph is already tiny (95 ns per kernel pair, 42 us of 4884 us). PDL only pays by hiding the *ramp* of the next kernel, and only if **both** kernels of a pair are PDL-aware. Measured: 0.15-0.3 us per small kernel pair, about 0.8 us per skinny GEMM that prefetches its weights into L2 before `griddepcontrol.wait`. FlashInfer's MoE and XQA kernels already use PDL (37% of the boundaries in the graph overlap today). | realistic **50-150 us (about 1-2%)**, ceiling about 210 us (3%) if every kernel were PDL. About 0 at c>=16 (steps of 14-29 ms). | Our GEMMs alone give nothing (+0.3%). The gain needs the neighbours too: inductor kernels (there is a switch, but see below, untested e2e) and our own GEMMs. |
| **B. swap-AB / smaller token tile** | FlashInfer's CUTLASS MoE **already is swap-AB** (decode tactics 16/17 and 56/57 = 128x32 tile, weights on the MMA M side, tokens on N=32). Shrinking the token tile further would save at most **3.3 us/layer = 0.13 ms/step (2%)** (extrapolated from the measured tile-size slope). The MoE decode kernel is byte-bound, not padding-bound. | tile shrink: <= 130 us. **But** b12x's small-batch kernels (CUDA-core `direct_micro` for M<=3, MMA `micro` for M<=5, no padding, no glue launches) are faster than CUTLASS with cold weights: **-34% at M=3, -16% at the padded c=1 size M=4 (38.7 -> 32.4 us/layer = 0.25 ms per step), -13% at M=8 (c=2, 0.32 ms per step)**, equal from M=12. | Not a new tile: a fused small-M MoE path. It already exists in FlashInfer; blockers are the activation-scale convention (kernels/moe-nvfp4 4c: use unit scales) and an end-to-end quality check. |

Recommended next step: section "Next step" at the end.

---------------------------------------------------------------------------------------------------

## A. Programmatic dependent launch

### A.1 Step 1: how big is the gap inside the decode graph? (no GPU, existing traces)

Source: `kernels/decode-roofline/traces/node_c1.sqlite` and `node_c64.sqlite` (nsys `--cuda-graph-trace=node`,
`CONFIG=optimized`, no skinny patch). Scripts: `gap_analysis.py`, `dur_hist.py`, `overlap_pairs.py`,
`pdl_candidates.py`, `pdl_edges.py`; outputs in `results/`. The node-level span of the target graph (4884 us)
matches the graph-level trace (4875 us), so node tracing does not inflate the inside of the graph.
Graph memset nodes are counted as nodes (41 per step, they are the MoE output zeroing).

Target verify graph, c=1 (323 launches, 1072 nodes = 1031 kernels + 41 memset/memcpy):

| metric | value |
|---|---|
| graph span (median) | 4884 us |
| sum of node durations | 5548 us (more than the span: the side stream and PDL overlap) |
| GPU idle inside the graph (span minus union of nodes) | **42 us (0.9%)** |
| boundaries (same-stream consecutive nodes) | 1070 per step |
| boundaries with a gap below 2 us | **all of them** (p50 0 ns, p90 96 ns, p99 320 ns) |
| boundaries that already overlap (= PDL in effect) | **400 of 1070 (37%)** |
| typical gap on a non-overlapped boundary | **95-100 ns** (the CUDA-graph kernel-to-kernel floor) |
| kernels shorter than 1.5 / 2.5 / 4 us | 300 / 483 / 758 (324 / 678 / 1526 us in total) |
| MTP drafter graphs (38 and 30 nodes) | gaps 6.6 and 5.6 us of 545 and 470 us |
| step-level idle outside the graphs (host/scheduler, from the decode-roofline README) | 145 us (skinny) to 644 us (base) per step, not addressable by PDL |

c=64 (109 launches, 1302 nodes): gap total **74 us of 24749 us (0.3%)**. PDL is irrelevant there.

The three largest per-pair gaps (us per step = pairs x mean gap): inductor quant kernel -> FP8 GEMM 11.2 us
(40 x 280 ns), `kernel_mha` (XQA) -> inductor 4.4 us (10 x 444 ns), and memset node boundaries 3.8 + 3.8 us
(`doActivation` -> memset -> MoE GEMM2, 40 x 95 ns each; the memset itself is 0.77 us x 40 = 31 us per step
and cannot be PDL-overlapped; zeroing the output inside an earlier kernel would remove it, but that is a
FlashInfer change). Everything else is the ~95 ns floor:

```
 3.8 us x40 95 ns  fp8_gemm -> triton(add+rmsnorm)       3.0 us x30 98 ns  bf16 wmma gemm -> triton(zeros)
 3.8 us x39 97 ns  moe grouped gemm -> triton(quant)      2.9 us x30 98 ns  conv1d -> gdn_decode
 3.8 us x40 94 ns  triton(quant) -> fp8_gemm              2.8 us x30 95 ns  gdn_decode -> triton(quant)
```

Who is already PDL (consumer side, `results/pdl_candidates_node_c1.txt`): MoE routing/permute glue 80% of its
200 boundaries overlap (365 ns average overlap), CUTLASS grouped GEMM 75% (436 ns), XQA 100%, conv1d 97%
(vLLM Triton `launch_pdl`), `cvt_fp16_to_fp4` 67%. FlashInfer passes `enable_pdl` = `device_support_pdl()` =
`major >= 9`, which is true on SM120, and the SM120 grouped-GEMM kernel (`sm120_gemm_tma_warpspecialized_*`)
has `wait_on_dependent_grids` / `launch_dependent_grids`. **PDL is therefore already on for the MoE chain and
for XQA.** Not PDL today: the 190 inductor Triton kernels (norm/quant/act), the 80 dense FP8 GEMMs (vLLM
CUTLASS `_C`), the 69 BF16 router/in_proj_ba GEMMs (cuBLAS), GDN decode, rope/kv-write, ATen elementwise, gemv.

### A.2 Step 2: what PDL can recover

**Microbenchmark** (`pdl_bench/pdl_chain.cu`, built with `build.sh`, run by `run_pdl_gpu_v2.sh` under the lock,
host process, no container). A CUDA graph of 160 dependent kernels: 16 layers of
`glue(amax+fp8 quant) -> skinny FP8 GEMM 12288x2048 -> glue -> 6 tiny 1-2 us kernels -> skinny FP8 GEMM
2048x4096`, M=4, 538 MB of distinct weights (4x the L2, so cold DRAM reads like a real step). The GEMM is a copy of
`kernels/dense-fp8-gemm/skinny/skinny_fp8.cu` with the PDL hooks. Launch attribute:
`cudaLaunchAttributeProgrammaticStreamSerialization` through `cudaLaunchKernelEx`, captured into the graph.
Inside kernels `griddepcontrol.wait` / `launch_dependents` (PTX). All modes give a **bitwise identical output**
(checksum column in the log). Median of 36 samples x 100 graph replays, two full runs agree to 0.1%.

| config (graph of 160 kernels, M=4) | us / graph | vs plain |
|---|---:|---:|
| plain | 473.0 | |
| our GEMMs PDL (wait + trigger), neighbours plain | 474.6 | +0.3% |
| our GEMMs PDL + weights prefetched to L2 before the wait, neighbours plain | 497.5 | **+5.2%** (worse) |
| glue/tiny kernels PDL only, GEMMs plain | 453.4 | **-4.2%** (-19.7 us) |
| everything PDL (wait + trigger) | 458.2 | -3.1% (-14.9 us) |
| everything PDL + weights prefetched to L2 (`cp.async.bulk.prefetch.L2`) before the wait | **428.0** | **-9.5% (-45 us)** |
| (M=16, same chain) plain / everything PDL + L2 prefetch | 476.1 / 445.0 | -6.5% |

Floor (600 dependent 4-CTA kernels): 0.693 us/kernel plain, **0.512 us/kernel PDL (-0.18 us each)**.

nsys node trace of the microbenchmark (`results/v2_chain_gap_summary.txt`): plain has **94 ns of gap** per
kernel pair (the same ~95 ns as in the real decode graph, 3.0% of span); with PDL the gap drops to 11-12 ns and
58 of 59 pairs overlap, but the span falls from 182.3 to 163.7 us (-18.6 us, 0.31 us per kernel). Only about
a third of that is the visible gap. The rest is the ramp of the next kernel, which PDL hides.

Reading:

* A PDL edge needs **both ends**: the consumer launched with the attribute *and* the producer calling
  `launch_dependents` early. Our GEMMs with plain neighbours get nothing (+0.3%), and prefetching weights
  there only adds traffic (+5%). So "launch our skinny kernels with PDL" is, on its own, worth 0.
* Per boundary: 0.15-0.18 us for tiny kernels, about 0.8 us per GEMM when the GEMM prefetches its weights into
  L2 while the producer runs ((45 - 19.7) us / 32 GEMMs). Prefetching registers (`ldg` before the wait) was
  not faster than the plain wait, and it exposed a real hazard: with `ld.global.nc` for the activations, ptxas
  hoisted the loads above `griddepcontrol.wait` and the output was wrong in one config (v1 run, kept in
  `results/v1_hoisting_bug/`). Activation loads after a wait must be plain `ld.global` with a memory clobber.
* **Inductor switch.** torch 2.13 has `TORCHINDUCTOR_ENABLE_PDL=1`: it emits `gdc_wait()` /
  `gdc_launch_dependents()` and `launch_pdl=True` into the generated Triton kernels (checked in the cache,
  `results/inductor_pdl_codegen_check.txt`: 3 of 3 kernels have it with the flag, 0 without). A test with
  inductor kernels alternating with cuBLAS GEMMs (`pdl_inductor.py`, 12 layers, M=4) gave **125.0 us with and
  without the flag, 0 overlapping kernel pairs**: exactly the "one end only" case above. The flag will only
  help where two PDL kernels are adjacent or the neighbouring GEMM is PDL-aware too. It has not been tested
  in the server (vLLM's compile cache key must also change; use a fresh `VLLM_CACHE_ROOT`).

**Upper bound and realistic estimate for the c=1 step.** `pdl_edges.py` counts the boundaries where both
kernels would be PDL-aware and that do not overlap today (per step of 1070 boundaries, 670 non-overlapped):

| scenario | new PDL edges / step | estimate |
|---|---:|---:|
| inductor flag only | 90 | 90 x 0.15 = ~14 us |
| our skinny GEMMs PDL-aware only | 70 | ~0 (producer not aware) |
| inductor flag + our GEMMs (FP8 + BF16, ~150 GEMM consumers with L2 prefetch) | 349 | 150 x 0.8 + 200 x 0.15 = **~150 us** |
| everything PDL (also cuBLAS, GDN decode, rope, gemv, ATen) | 589 | ~200 us (ceiling: 670 x 0.31 = 208 us) |

That is **1-2% of the 6.4 ms c=1 step realistically (up to 3% as a ceiling), and about 0 at c>=16.**
The estimate uses microbenchmark costs per boundary, not a model run; it is a bound, not a measurement.
The kernels we do not own (cuBLAS wmma/gemv, FlashInfer, vLLM `_C` quant, Triton from vLLM's own ops) cannot be
switched; the realistic scope is: inductor flag + our two skinny GEMMs (the FP8 one is CUDA, the BF16 one is
Triton, which supports `launch_pdl` and `tl.extra.cuda.gdc_wait`).

---------------------------------------------------------------------------------------------------

## B. Swap-AB / token tile for the NVFP4 MoE at decode

### B.1 What the two backends do at 1-24 tokens (source read in the vllm/vllm-openai:v0.32.0 image)

Model: 256 experts, top-8, hidden 2048, moe_intermediate 512. c sequences x 3 verified tokens (MTP-2). vLLM
pads the decode batch to the next CUDA-graph size (1 2 4 8 16 24 32 ...), so the MoE layer sees M = 4 for c=1,
8 for c=2, 16 for c=4, 24 for c=8, 48 for c=16 (the padded rows are routed too).

* **FlashInfer CUTLASS (the default, `FLASHINFER_CUTLASS`)**: `cutlass_fused_moe` -> `trtllm::fused_moe::gemm1/gemm2`.
  The SM120 grouped GEMM is a persistent warp-specialized kernel (grid 1x188, 384 threads, 87-93 KB smem). Tile
  table (`get_candidate_configs_sm120` x swapAB x finalize): 10 CTA tiles from 128x32x64B to 256x128x128B. Autotuned decode
  picks (`results/tactics_sweep_L19.txt`): GEMM1 tactic 16/17 = **128x32 + swapAB**, GEMM2 56/57 = 128x32 +
  finalize + swapAB. With `SwapAB=true` the launcher makes the weights the MMA A operand
  (`MainloopElementA = Weight`), so the **128-wide tile M is the weight rows and the 32-wide N is the token
  slot of one expert**. That is DeepGEMM's sm100 small-M choice. Minimum token tile in the candidate set: 32.
* **FlashInfer b12x (`--moe-backend flashinfer_b12x`)** puts tokens on M (no swap). `select_sm120_moe_backend`
  / `launch_sm120_moe`: routed rows < 32 and M <= 8 (i.e. **M <= 3 at top-8**) -> `direct_micro`, a CUDA-core
  FP4 dot-product kernel (`fp4_dot8_sum_f32acc`), no MMA, no padding, routing and quant fused in the same
  kernel, only for intermediate size <= 512 (this model: 512); rows <= 40 (M <= 5) -> MMA `micro`; rows <= 1024
  -> `static` (M tile 64 with a 32-row virtual-expert split, so at most 32 valid rows per tile); above that
  `dynamic` with M tile 16/32/64/128 chosen from rows per expert (`_select_dynamic_tile_m`).

### B.2 How much of the MMA work is padding

Measured routing (real layer 19 and the captured text, `tactics_sweep_L19.txt`): tokens per **active** expert
M=3: 24 rows / 23 active = 1.04; M=12: 96 / 70 = 1.4; M=48: 384 / 131 = 2.9; M=192: 1536 / 201 = 7.6.
With a 32-wide token tile the useful part of the MMA N dimension is **3%, 4%, 9%, 24%**. The rest is padding.

Byte/FLOP argument per layer at M=3 (23 experts x 3 x 512 x 2048 FP4 params):

* weights 23 x 1.77 MB = 41 MB -> 25 us at 1.6 TB/s;
* padded MMA 23 x 201 MFLOP = 4.6 GFLOP; useful 0.15 GFLOP. At the measured 2.2 PFLOPS NVFP4 peak the padded work is
  2.1 us, the useful 0.07 us. Arithmetic intensity is 3.5 FLOP/B per token per expert against a ridge of
  1200 FLOP/B, so the kernel needs about 340 tokens per expert to become compute bound;
* the kernel does **not** wait for the MMAs: the tactic sweep says that making the token tile 4x wider
  (128x32 -> 128x128, 4x the MMA work) moves GEMM1 by +5.6 us at M=3 (43.2 -> 48.8 us, default GEMM2) but by only
  **about +1% at M=192** (282.6 -> 286.3 us) and +0.2% at M=48 (182.5 -> 182.8 us), where bytes dominate.
  These sweeps replay one layer, so the weights are warm in L2; with cold weights the MMA share can only be smaller.

Extrapolating the M=3 slope backwards (0.06 us per padded token column for GEMM1, 0.08 for GEMM2) to a token tile
of 8 (the m16n8k64 minimum) saves about 1.4 + 1.9 = **3.3 us per layer, 130 us per step (2.7% of the 4.9 ms verify
graph)**. A tile of 16 saves about half. It is an upper bound for "same kernel, narrower tile", and it needs a
rebuilt `fused_moe_120` module (CUTLASS template change), since 128x32 is the smallest instantiated tile.

### B.3 Fraction of the MoE decode time

In-graph (c=1, node trace, `decode-roofline` categories): grouped GEMM1/2 = 1227 us per step (18.7 + 11.9 us per
layer, 40 layers), routing/permute glue 760 us (8 launch-bound kernels per layer), shared expert 306 us.
Bytes for the 21-23 distinct experts of the bytes model (37-41 MB, 23-25 us at 1.6 TB/s) against 30.6 us
measured: the grouped GEMMs run at 75-83% of the byte floor, so the headroom of **any** better grouped GEMM is
5-8 us/layer = 0.2-0.3 ms/step. Padding is ~3.3 us of
that, the rest is pipeline fill (8 k-iterations per tile), tail and launch ramp.

### B.4 Cold-weight measurement: CUTLASS swap-AB vs b12x

The existing microbenchmarks (`kernels/moe-nvfp4`) replay one layer, so its 41 MB of active weights sit in the
128 MB L2. `bench_cold_moe.py` (`run_cold_moe_gpu.sh`) calls 5 real layers (0, 9, 19, 29, 39) in one CUDA graph
with a 300 MB memset between calls (flush-only graph subtracted), real captured hidden states, real routing,
consecutive tokens in groups of 3 (what a verify step feeds). Both include their input quantization, router
excluded. b12x runs with unit activation scales (the only setting that is right for every M, moe-nvfp4 4c).
us per layer, cold:

| M (decode batch) | distinct experts / layer | bytes MB | byte floor @1.6 TB/s | CUTLASS (swap-AB 128x32) | b12x | b12x / CUTLASS | b12x saves per step (x40 layers) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 (c=1, unpadded) | 18.6 | 33 | 21 | 33.5 | **22.0** | **0.66** | 456 us |
| **4 (c=1 as vLLM runs it)** | 22.4 | 40 | 25 | 38.7 | **32.4** | **0.84** | **252 us** |
| 6 | 31.0 | 55 | 34 | 49.5 | 43.3 | 0.87 | |
| **8 (c=2)** | 40.2 | 71 | 45 | 60.8 | **52.7** | **0.87** | **324 us** |
| 12 | 57.0 | 101 | 63 | 79.5 | 82.7 | 1.04 | |
| 16 (c=4) | 68.4 | 121 | 76 | 92.8 | 90.3 | 0.97 | 100 us |
| 24 (c=8) | 86.4 | 153 | 96 | 115.4 | 112.2 | 0.97 | 130 us |
| 48 (c=16) | 136.4 | 241 | 151 | 174.1 | 170.9 | 0.98 | 130 us |

(M=3, 4, 8, 16, 24, 48 from `cold_moe_v3_padded.txt`, M=6 and 12 from `cold_moe_v2.txt`; the two runs repeat M=3, 24, 48
within 3%. Differences of 3% (about 3 us per layer) at M >= 16 are at the noise level.) The 4th token at M=4 is a
real token here, not the zero-hidden padding row vLLM adds; the padding row is routed too, so the real expert count
at c=1 is likely a bit higher than 22.

* At M=3 b12x (direct_micro) reaches 1.5 TB/s end to end, i.e. **the byte floor**, while CUTLASS is at 1.0 TB/s.
  The gain does not come from the tile shape but from a different design: no padding, no permute / activation /
  strides launches, one fused kernel. But vLLM runs c=1 at M=4 (routed rows 32, not < 32), where b12x switches to
  its MMA `micro` kernel: still **16% faster (6.3 us/layer = 0.25 ms per c=1 step, 4% of ITL)**; at M=8 (c=2) 13%
  faster (**0.32 ms per step**). The 2.5-3 us advantage at M >= 16 is within noise.
* From M=12 up both are 20-30% above the byte floor and equal. No tile shape would change that; the gap to the
  floor is fill/drain and launch.
* Numerics: b12x with unit activation scales differs from CUTLASS by rel-L2 0.06-0.16 per layer. The earlier
  report (moe-nvfp4 4b) shows both at 0.136 against an fp32 reference on layer 19 at 4096 tokens, so the difference is
  FP4-grid noise and not an error, but it was **not** checked at M <= 4, and it replaces the checkpoint's calibrated
  static activation scales by dynamic ones. It needs the NLL / MTP-acceptance check before anything is shipped.

---------------------------------------------------------------------------------------------------

## Next step

1. **B first, and it is a dispatch change, not a kernel**: in the existing hybrid patch (`kernels/moe-nvfp4`,
   `VLLM_B12X_HYBRID_MIN_TOKENS`, which now sends every batch below 1024 tokens to CUTLASS) route the padded decode
   sizes M = 4 and 8 (c = 1, 2) to b12x with unit activation scales and keep CUTLASS for 9..1023. Then NLL, needles and MTP
   acceptance (dynamic scales replace the calibrated ones, rel-L2 vs CUTLASS 0.06-0.16, parity with CUTLASS against
   fp32 only shown at 4096 tokens), and a decode A/B at c=1 and 2 (`run_ab.sh` style, fresh server per side).
   Expected: 0.25-0.3 ms of a 6.4-7 ms step (4%). Cheap to try; the risk is numerics, not speed.
2. Do not write a swap-AB / narrower-tile kernel for the CUTLASS grouped GEMM: at most 0.13 ms/step, needs a
   FlashInfer rebuild, and the b12x path already removes the padding and the glue.
3. PDL: only worth doing as a side effect. If the skinny FP8/BF16 GEMMs get a second revision, add the
   `launch_pdl`/`griddepcontrol` hooks and the L2 weight prefetch, and test `TORCHINDUCTOR_ENABLE_PDL=1` in the server
   (fresh compile cache) as a one-line A/B. Expect <= 1-2% at c=1 and nothing at c>=4. Do not expect gains from adding
   the attribute to our GEMMs alone.
4. Cheap find, not pursued: the 41 memset nodes (0.77 us each + 2 gaps) around MoE GEMM2 cost about 35 us per step
   and are not PDL-able; zeroing the output in `doActivation` would remove them (FlashInfer change).

## Files

* `gap_analysis.py` (idle gap per kernel pair, top neighbour pairs), `dur_hist.py` (duration histogram, union idle),
  `overlap_pairs.py` (which pairs overlap today), `pdl_candidates.py` (who controls each consumer),
  `pdl_edges.py` (PDL-edge scenarios). CPU only. Usage: `python3 -I <script> <node_trace.sqlite>`.
* `pdl_bench/pdl_chain.cu` (v2), `pdl_chain_v1.cu` (has the `ld.global.nc` hoisting hazard), `build.sh`,
  `run_pdl_gpu_v2.sh` (host process under the lock, ~1 minute of GPU), `pdl_inductor.py`, `run_inductor_gpu_v2.sh` (the PDL=0 vs 1 inductor test; v1 of it is `results/pdl_inductor_v1.txt`).
* `bench_cold_moe.py`, `run_cold_moe_gpu.sh` (uses a private copy of the FlashInfer JIT cache).
* `results/`: raw logs. `v2_*` is the valid PDL run, `v1_hoisting_bug/` the first run (all configs except
  "all PDL + weight regs" agree with v2), `cold_moe_v2.txt` (M=3,6,12,24,48) and `cold_moe_v3_padded.txt` (M=3,4,8,16,24,48) the MoE runs.

Reproduce: `./build.sh && ./run_pdl_gpu_v2.sh`; `./run_cold_moe_gpu.sh 3 4 8 16 24 48`; CPU analysis as above.
Limits: the PDL chain uses synthetic weights and an amax-quant stand-in for the Triton glue, one GPU, one clock
state (2850 MHz idle at start); the decode-graph gap numbers come from a node-level trace of the base config
(not the skinny patches); nothing was run end to end.

## Focused pinned PDL pair — 2026-10-28

**Not selected.** Both controls completed six prefill cases, four greedy decode cases and five rounds of ten cold NLL probes (exit0/cleanup0). The strict comparator verified the same18 constructor/first-launch identities and singleton configs, plus12 model-bound identities. OFF binds96 serialized autotuners+107 bundled artifacts; ON binds111 bundled artifacts. Both FI shutdown checks retain132 fixed-reference entries and source checks pass.

An external read-only import in another container initialized CUDA during ON. Root reports no explicit GPU kernels/allocation requested, but exact start is uncertain; the conservative04:19–04:21UTC window overlaps c16/c64 measured commands and NLL. All six prefill and c1/c4 finish before that window. Aggregate GPU telemetry cannot assign activity to that context. The pair is not a clean dedicated-GPU comparison.

| Scenario | Metric | OFF ms | ON ms | ON change | Scope |
|---|---|---:|---:|---:|---|
| prefill-2048 | median TTFT | 64.145 | 63.461 | -1.07% | outside reported context window; single pair |
| prefill-4096 | median TTFT | 99.231 | 95.954 | -3.30% | outside reported context window; single pair |
| prefill-5000 | median TTFT | 112.847 | 112.554 | -0.26% | outside reported context window; single pair |
| prefill-8192 | median TTFT | 186.483 | 181.815 | -2.50% | outside reported context window; single pair |
| prefill-16384 | median TTFT | 359.225 | 348.944 | -2.86% | outside reported context window; single pair |
| prefill-32768 | median TTFT | 801.622 | 794.258 | -0.92% | outside reported context window; single pair |
| decode-c1 | median TPOT | 2.099 | 2.113 | +0.67% | outside reported context window; single pair |
| decode-c4 | median TPOT | 2.888 | 3.012 | +4.28% | outside reported context window; single pair |
| decode-c16 | median TPOT | 5.453 | 5.633 | +3.29% | possible external-context overlap |
| decode-c64 | median TPOT | 11.644 | 11.789 | +1.24% | possible external-context overlap |

Observed prefill reductions are modest (0.26–3.30%) and require an independent reversed-order repeat before any benefit claim. Decode c1/c4 regress0.67%/4.28% in this pair; the other decode cases are flagged above. Do not infer actual producer-consumer overlap from these timings.

NLL remains unresolved even with compiler configurations pinned. Maximum within-control range: OFF0.085944, ON0.156225nat/token; logprobs-1 mean shifts+0.029119 and logprobs-2+0.009721. All10 prompts exceed the preregistered within-mode0.001 investigation trigger and7 means exceed its absolute-shift trigger. The policy reports `investigate_numerical_change`, not numerical/quality equivalence. ON NLL also overlaps the conservative external-context window.

Evidence: `results/pdl_focused_pair_1.json` (compiler proof plus explicit performance-validity qualifier), preserved original comparator output `pdl_focused_pair_1.compiler_verified_original.json`, per-control `pin_aot_verify.json`, and `pdl_focused_pin1_1/external_context_contamination.json` with exact command intervals and GPU samples. No reference expansion occurred. No stack or agentic run was started.
