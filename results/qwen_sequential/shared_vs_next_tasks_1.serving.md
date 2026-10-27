Baseline `next_tasks_1`; candidate `b12x_tasks_1`; GPU utilization 0.78.

Positive rate changes mean faster; negative latency changes mean faster.

**throughput**

| task | total tok/s baseline / candidate | change | TTFT p50 ms baseline / candidate | change | ITL p50 ms baseline / candidate | output tok/s change | MTP acceptance baseline / candidate |
|---|---:|---:|---:|---:|---:|---:|---:|
| long-doc | 33256.7 / 33792.3 | +1.6% | 1239.4 / 1223.4 | -1.3% | 7.18 / 7.24 | +1.6% | 37.0% / 36.8% |
| prefill-heavy | 41631.5 / 42760.9 | +2.7% | 326.2 / 345.4 | +5.9% | 10.29 / 10.22 | +2.7% | 30.5% / 32.6% |
| rag | 23202.2 / 23476.4 | +1.2% | 169.9 / 184.5 | +8.6% | 14.29 / 14.44 | +1.2% | 51.2% / 51.4% |
| shared-prefix | 27445.1 / 28812.7 | +5.0% | 219.5 / 226.5 | +3.2% | 14.34 / 14.28 | +5.0% | 56.9% / 60.2% |
| mixed-len | 18918.8 / 19898.9 | +5.2% | 232.7 / 207.2 | -10.9% | 20.54 / 20.58 | +5.2% | 65.1% / 62.2% |
| chat | 8157.3 / 8172.9 | +0.2% | 154.8 / 154.1 | -0.4% | 19.87 / 19.85 | +0.2% | 62.6% / 61.1% |
| decode-heavy | 4567.5 / 4623.7 | +1.2% | 175.2 / 153.9 | -12.1% | 29.13 / 29.24 | +1.2% | 64.0% / 62.8% |

**prefill**

| task | total tok/s baseline / candidate | change | TTFT p50 ms baseline / candidate | change | ITL p50 ms baseline / candidate | output tok/s change | MTP acceptance baseline / candidate |
|---|---:|---:|---:|---:|---:|---:|---:|
| prefill-2048 | 30691.5 / 31557.8 | +2.8% | 66.2 / 65.7 | -0.8% | 0.00 / 0.00 | +2.8% | — / — |
| prefill-4096 | 38058.3 / 40947.7 | +7.6% | 106.9 / 98.8 | -7.6% | 0.00 / 0.00 | +7.6% | — / — |
| prefill-5000 | 40994.3 / 43153.6 | +5.3% | 122.6 / 116.2 | -5.2% | 0.00 / 0.00 | +5.3% | — / — |
| prefill-8192 | 42977.8 / 49598.9 | +15.4% | 191.6 / 164.3 | -14.3% | 0.00 / 0.00 | +15.4% | — / — |
| prefill-16384 | 42479.0 / 49640.1 | +16.9% | 384.5 / 329.5 | -14.3% | 0.00 / 0.00 | +16.9% | — / — |
| prefill-32768 | 37755.2 / 43228.0 | +14.5% | 866.3 / 756.6 | -12.7% | 0.00 / 0.00 | +14.5% | — / — |

**decode**

| task | total tok/s baseline / candidate | change | TTFT p50 ms baseline / candidate | change | ITL p50 ms baseline / candidate | output tok/s change | MTP acceptance baseline / candidate |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode-c1 | 348.9 / 385.0 | +10.3% | 55.8 / 42.2 | -24.5% | 6.52 / 6.56 | +10.3% | 44.1% / 53.5% |
| decode-c4 | 1161.7 / 1127.2 | -3.0% | 62.1 / 61.4 | -1.0% | 8.21 / 8.32 | -3.0% | 64.1% / 61.1% |
| decode-c16 | 2415.3 / 2469.4 | +2.2% | 92.1 / 97.3 | +5.7% | 14.16 / 14.18 | +2.2% | 57.9% / 57.8% |
| decode-c64 | 5105.2 / 5019.6 | -1.7% | 123.8 / 130.0 | +5.0% | 29.10 / 28.99 | -1.7% | 59.5% / 60.5% |

These are single-run task comparisons. Dataset seeds fix prompts; sampling follows the request/model settings recorded with each run.
