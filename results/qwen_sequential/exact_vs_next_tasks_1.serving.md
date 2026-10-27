Baseline `next_tasks_1`; candidate `exact_tasks_1`; GPU utilization 0.78.

Positive rate changes mean faster; negative latency changes mean faster.

**throughput**

| task | total tok/s baseline / candidate | change | TTFT p50 ms baseline / candidate | change | ITL p50 ms baseline / candidate | output tok/s change | MTP acceptance baseline / candidate |
|---|---:|---:|---:|---:|---:|---:|---:|
| long-doc | 33256.7 / 36262.4 | +9.0% | 1239.4 / 1183.2 | -4.5% | 7.18 / 7.25 | +9.0% | 37.0% / 52.8% |
| prefill-heavy | 41631.5 / 42786.3 | +2.8% | 326.2 / 381.6 | +17.0% | 10.29 / 10.08 | +2.8% | 30.5% / 29.5% |
| rag | 23202.2 / 23891.6 | +3.0% | 169.9 / 176.3 | +3.7% | 14.29 / 14.36 | +3.0% | 51.2% / 54.4% |
| shared-prefix | 27445.1 / 27587.6 | +0.5% | 219.5 / 230.8 | +5.2% | 14.34 / 14.09 | +0.5% | 56.9% / 53.7% |
| mixed-len | 18918.8 / 19020.7 | +0.5% | 232.7 / 202.2 | -13.1% | 20.54 / 20.49 | +0.5% | 65.1% / 60.6% |
| chat | 8157.3 / 8272.3 | +1.4% | 154.8 / 140.6 | -9.2% | 19.87 / 19.68 | +1.4% | 62.6% / 61.8% |
| decode-heavy | 4567.5 / 4593.7 | +0.6% | 175.2 / 175.2 | +0.0% | 29.13 / 29.05 | +0.6% | 64.0% / 62.0% |

**prefill**

| task | total tok/s baseline / candidate | change | TTFT p50 ms baseline / candidate | change | ITL p50 ms baseline / candidate | output tok/s change | MTP acceptance baseline / candidate |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefill-2048 | 30691.5 / 27419.5 | -10.7% | 66.2 / 74.1 | +11.9% | 0.00 / 0.00 | -10.7% | — / — |
| prefill-4096 | 38058.3 / 39297.6 | +3.3% | 106.9 / 102.0 | -4.6% | 0.00 / 0.00 | +3.3% | — / — |
| prefill-5000 | 40994.3 / 42602.5 | +3.9% | 122.6 / 116.4 | -5.1% | 0.00 / 0.00 | +3.9% | — / — |
| prefill-8192 | 42977.8 / 43412.0 | +1.0% | 191.6 / 187.3 | -2.3% | 0.00 / 0.00 | +1.0% | — / — |
| prefill-16384 | 42479.0 / 44723.6 | +5.3% | 384.5 / 363.3 | -5.5% | 0.00 / 0.00 | +5.3% | — / — |
| prefill-32768 | 37755.2 / 40512.6 | +7.3% | 866.3 / 809.3 | -6.6% | 0.00 / 0.00 | +7.3% | — / — |

**decode**

| task | total tok/s baseline / candidate | change | TTFT p50 ms baseline / candidate | change | ITL p50 ms baseline / candidate | output tok/s change | MTP acceptance baseline / candidate |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode-c1 | 348.9 / 393.9 | +12.9% | 55.8 / 56.2 | +0.6% | 6.52 / 6.52 | +12.9% | 44.1% / 56.1% |
| decode-c4 | 1161.7 / 1191.0 | +2.5% | 62.1 / 86.6 | +39.6% | 8.21 / 8.26 | +2.5% | 64.1% / 65.0% |
| decode-c16 | 2415.3 / 2436.3 | +0.9% | 92.1 / 107.4 | +16.6% | 14.16 / 14.10 | +0.9% | 57.9% / 56.7% |
| decode-c64 | 5105.2 / 4843.1 | -5.1% | 123.8 / 147.4 | +19.0% | 29.10 / 28.61 | -5.1% | 59.5% / 61.1% |

These are single-run task comparisons. Dataset seeds fix prompts; sampling follows the request/model settings recorded with each run.
