# Scalar L19/M1024: first successful counter collection

Updated 2026-10-28 after the user supplied `scalar_l19_m1024_counters_2`.
The report was imported offline with CUDA hidden; no new GPU run was launched.
This supersedes the *current* permission-blocked status, while preserving the
failed first attempt as history. No new serving patch is promoted.

## Evidence and scope

The [raw report](../results/prefill_counter_study_2/counters.ncu-rep),
[CSV export](../results/prefill_counter_study_2/counter_table.csv),
[summary and calculations](../results/prefill_counter_study_2/summary.json), and
[artifact hashes](../results/prefill_counter_study_2/manifest.json) are published.
The original local directory is EXP-relative
`kernels/prefill-counter-study/results/scalar_l19_m1024_counters_2/`.

NCU contains one selected `gatedMoEGatedDynamicKernel` record on SM120, grid
`(1,1,188)`, block `(288,1,1)`. This is the native fused kernel inside the complete
scalar expert call, not all surrounding kernels or a serving request. Application
replay was strict with cache-control/clock-control both `none`, after warmup.
The profiler process exited 0. All nine replay application reports passed their
exploratory repeat-output checks and agree on native source, checkpoint identity,
capture hash, layer19, M1024 and tileM128. Replays are not independent benchmark
trials, and these output checks are not FP32 or model-quality equivalence.

| Quantity | Imported value |
|---|---:|
| GPU kernel duration | 311.680 µs |
| DRAM byte rate, aggregate | 1.402691 TB/s |
| L2 sector hit rate | 60.387926% |
| L2 throughput | 34.245851% of NCU peak sustained |
| Tensor pipeline activity | 20.870297% of NCU peak sustained |
| SM throughput | 20.870297% of NCU peak sustained |
| Registers per thread | 168 |
| Shared memory allocated per block, including driver allocation | 102,400 bytes = 100 KiB |
| Residency limits from registers / shared memory | 1 / 1 block per SM |
| Waves per SM | 1 |

## What the numbers support

**DRAM traffic/throughput is the primary optimization target for this isolated
warmed case.** Against the user-supplied device ceiling of 1.79 TB/s, the observed
rate is 78.36%. The much lower compute and L2 throughput percentages support this
classification. They do not prove an exclusive limiter or rule out load latency,
tail imbalance, barriers or overhead. The 1.79 TB/s reference is not a DRAM peak
metric collected in this report; sustained attainable bandwidth needs its own
matched characterization if a precise ceiling is required.

For a constant byte count, the ideal calculation is:

```text
Aggregate DRAM traffic ≈ 1.402691 TB/s × 311.680 µs = 437.191 MB
Ideal time at 1.79 TB/s ≈ 244.241 µs
Ideal time reduction ≈ 21.64%; ideal speedup ≈ 1.276×
```

Thus ~22% is the ideal remaining time reduction **at unchanged traffic**, not
a universal optimization limit. Better memory access scheduling might improve
achieved bandwidth; reducing bytes can go beyond this constant-traffic estimate.
Neither improvement is guaranteed. Kernel headroom is not serving headroom.

## Important interpretation corrections

- DRAM byte rate is aggregate traffic. The report does not split reads/writes,
  so 437 MB must not be described as measured weight reads alone.
- For E256, H2048, I512, packed FP4 FC1+FC2 weights occupy **384 MiB = 402.653 MB**.
  One E4M3 scale per 16 weight elements adds **48 MiB = 50.332 MB**, before global
  scales/metadata. Their combined 432 MiB = 452.985 MB exceeds the reported
  128 MiB L2 capacity. Similarity to measured traffic is suggestive, not a byte
  attribution: cache reuse, activations and other buffers also contribute.
- L2 hit rate is a sector statistic, not the percentage of expert weights cached.
  34% aggregate L2 throughput does not rule out every local cache bottleneck.
- Nine resident warps out of a maximum 48 gives an **18.75% residency ceiling**.
  Achieved occupancy over execution was not collected in this metric set.
- This is a persistent kernel. One wave describes the CTA grid, not the number
  of expert/tile tasks processed. Resident warps and pipeline stages can hide
  latency without a second wave. Raising occupancy is useful only if it improves
  actual duration without harmful traffic/resource changes.
- The raw CSV includes static device clock attributes, but no achieved-clock
  counters. Maximum device frequency is not proof that execution stayed there.
- Useful FC1+FC2 arithmetic is about 51.54 GFLOP, or **165.36 TFLOP/s** at the
  measured duration. This excludes other operations and padded MMA work; it is
  distinct from the tensor-pipeline activity metric and any separate GEMM peak.

These distinctions follow NVIDIA's [Compute Triage guidance](https://docs.nvidia.com/nsight-compute/ComputeTriage/index.html):
use a throughput breakdown and multiple indicators, and judge each change by
absolute duration. Counter collection here does not retrospectively prove why
every earlier tile/split/load-guard variant regressed.

## Collection route and audit limits

The user reports collecting this in a temporary container with `--gpus` and
`--cap-add=SYS_ADMIN`, using the study's `process_guard.py run`. This route resolves
counter access without rebooting, reloading the NVIDIA module, or recreating
`qwen-prefill-dev`. NVIDIA documents administrative privileges / `CAP_SYS_ADMIN`
as a supported route under restricted counter access.
[NVIDIA counter permission guidance](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-nsightcompute).

The imported report and process record verify successful collection. This second
directory does not contain the first launcher's complete source71/runtime13,
outer log and cleanup audit bundle; do not claim it passed that entire protocol.
Per-pass native-source/capture/checkpoint assertions are available. The source
hash is `993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445`.
Outside-range events are 781–854 µs under profiler injection and are excluded
from ordinary performance comparisons.

## Next decision

Start from the existing serving timeline: determine how much prefill elapsed time
this native kernel occupies and which M buckets actually matter. Then inspect
whether weights/scales are redundantly fetched across tiles and whether reuse
can improve without extra scratch or spills. Preserve selected experts and
weights; changing model routing is a quality intervention.

Choose one traffic/reuse or bandwidth-utilization hypothesis, compare complete
calls with independent unprofiled timing, and re-collect the same counter set to
check its mechanism. Do not broadly resume tile/occupancy sweeps from the 18.75%
figure alone. M4096, other layers and 40-layer serving cache behavior remain
unprofiled by this report. The decode Inductor-control gap remains separate.
