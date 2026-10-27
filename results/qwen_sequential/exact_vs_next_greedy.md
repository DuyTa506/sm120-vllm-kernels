Baseline `next_diagnostic_1`; candidate `exact_tasks_1`; explicit temperature 0; GPU utilization 0.78.

Prior suites: `diagnostic` / `individual`. Each case has its own warmup and measured cache salt.

| task | output tok/s base / candidate | change | TTFT p50 ms | ITL p50 ms | MTP acceptance | acceptance length | identical output texts |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode-c1 | 447.83 / 445.56 | -0.51% | 64.06 / 58.00 | 6.279 / 6.291 | 98.66% / 97.55% | 2.973 / 2.951 | 4/8 |
| decode-c4 | 1185.34 / 1202.00 | +1.41% | 82.02 / 92.31 | 7.930 / 7.825 | 83.89% / 88.28% | 2.678 / 2.766 | 4/24 |
| decode-c16 | 2583.60 / 2603.67 | +0.78% | 139.35 / 110.31 | 13.602 / 13.459 | 87.73% / 88.43% | 2.755 / 2.769 | 16/96 |
| decode-c64 | 4977.60 / 4941.52 | -0.73% | 164.01 / 170.84 | 27.301 / 27.483 | 87.01% / 86.80% | 2.740 / 2.736 | 64/384 |

MTP acceptance = accepted/drafted tokens; acceptance length = 1 + accepted tokens/draft steps. Both use measured-phase counter deltas. Matching generated texts is diagnostic, not a quality score.

Explicit temperature zero controls sampling. MTP acceptance, generated paths, startup autotuning and scheduling may still vary. This pair is not a confidence interval or a model-quality pass. Different prior suites are recorded as a preconditioning limitation.

The prior suites differ: preconditioning is not identical despite matching per-case warmups. Use this result as a diagnostic and repeat with matched run order/history before a causal timing claim.
