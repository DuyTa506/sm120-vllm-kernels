# CPU audit: prefill timing and runtime cache variation

The earlier shared and current shared_exact runs do **not** hold autotuning choices constant. The audit found 43 changed FlashInfer tactics among 132 common entries and four changed selected Inductor configurations, despite matching server arguments and generated code. These are concrete confounds; their timing contribution has not been measured. The two load-time alpha expressions do not establish a cause for the performance change.

CPU-only inspection of saved artifacts and cache text, captured at 2026-10-27T18:45:14.490356+00:00. No GPU jobs, HTTP requests, cache execution, or edits to active sources/artifacts. Full hashes, exact keys, selected configs, source excerpts and aligned telemetry are in [the JSON record](prefill_runtime_cpu_audit.json). This record embeds the cache snapshot so later cache changes do not erase the evidence.

Eight requests per row, concurrency 1, one output token; p50 TTFT in ms. The next comparison is the **earlier** next_tasks_1 run, not a contemporaneous repeated baseline.

| Input | Next | Shared | Shared exact | Exact vs shared latency | Shared latency reduction vs next | Exact latency reduction vs next |
|---:|---:|---:|---:|---:|---:|---:|
| 2048 | 66.23 | 65.68 | 74.09 | +12.80% | +0.83% | -11.86% |
| 4096 | 106.90 | 98.81 | 101.98 | +3.21% | +7.57% | +4.60% |
| 5000 | 122.63 | 116.21 | 116.44 | +0.20% | +5.24% | +5.05% |
| 8192 | 191.59 | 164.28 | 187.25 | +13.98% | +14.25% | +2.27% |
| 16384 | 384.54 | 329.50 | 363.34 | +10.27% | +14.32% | +5.51% |
| 32768 | 866.26 | 756.55 | 809.27 | +6.97% | +12.66% | +6.58% |

## Controls and concrete differences

- Shared and shared_exact have identical saved serve arguments and engine config. Their only environment difference is the isolated VLLM_CACHE_ROOT. Same image/checkpoint/GPU, max_num_batched_tokens=16384, MTP 2, TP/PP 1, BF16, async scheduling, graph sizes, chunked prefill, and GPU utilization 0.78. Shared/exact KV capacity both 3,437,190 tokens; next 3,369,930. Extra greedy decode and repeated-NLL phases follow the full matrix and do not change its commands.
- Cache factors match shared/exact. All 47 normalized generated Python files match as a content multiset, including the same backbone and eagle AOT keys. Opaque custom ops sit outside these graphs, so this is not proof of every historical backend source byte. Older runs predate source manifests.
- All three startup logs say 132 new FlashInfer configs, zero reused. The shared/exact differences include dense shared-expert gate/up GEMM M=8192, K=2048, N=1024 tactic 16→18; down GEMM M=8192, K=512, N=2048 tactic 8→14; and down GEMM M=16384, K=512, N=2048 tactic 23→15. Both M=2048 and M=4096 projection tactics match. Next uses 16/14 at 8192 and 24/15 at 16384. Installed source and checkpoint dimensions support mapping these dense FP4 keys to the shared expert; routed b12x experts use a separate path at these sizes. A 32768-token request is chunked at 16384, so the latter choice can matter. These cache choices do not prove their actual latency cost.
- Selected Inductor configs differ in two backbone and two eagle kernels. The backbone native BF16→FP8 row quantizer changes from R0_BLOCK=4096, 16 warps to 1024, 8 warps; next also uses 1024, 8. The backbone zero fill changes XBLOCK=512 / 8 warps → 1024 / 4. Eagle reduction and SiLU pointwise choices also differ. Identical generated source therefore does not ensure identical runtime kernels. Complete config hashes are recorded.

## Telemetry and interpretation limits

Result start_times are CLOCK_MONOTONIC; they were aligned to host-local CSV timestamps using the current wall-minus-monotonic offset and Asia/Ho_Chi_Minh timezone. The mapped windows agree with recorded CLI event windows. Samples inside individual measured requests are retained in the JSON. Telemetry is 1 Hz, often only 1–3 samples per short scenario, and utilization/power can lag.

At 8192, both shared and exact active-window samples report 2640 MHz. At 16384, sample median clocks are 2549.5 versus 2572 MHz; at 32768, 2561 versus 2580 MHz. This does not support an observed broad clock reduction, but cannot exclude fine-grained clock/power variation. Per-request TTFTs show a consistent shift at 8192 and 32768 rather than a single isolated outlier; only one eight-request run per variant is available.

Full matrix requests inherit temperature 1.0, top_k 20, top_p 0.95 because temperature is omitted. The prefill rows output one token, so long generated decode trajectories are not involved in their TTFT. Sampling/MTP variability remains material for the other scenarios. Sequential run order, host load and independent autotuning are uncontrolled. These results remain exploratory; neither the reduced gain versus the old baseline nor alpha rounding has a causal explanation from this audit.

For the later final A/B, share only supported common FlashInfer tactic keys and verify actual loading/cache hits/fallbacks; also capture selected Inductor config hashes and generated code hashes. Matching FlashInfer tactics alone does not make the executions globally identical. No change to the current queue is proposed.
