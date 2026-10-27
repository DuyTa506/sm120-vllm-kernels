# Shared b12x versus next: completed individual tasks

Reproduce locally (CPU only): `python3 results/qwen_sequential/compare_shared_vs_next_tasks_1.py`.

Baseline `next_tasks_1`; candidate `b12x_tasks_1` (`shared/`, before the exact-alpha correction). Candidate ran first, then baseline. Both exited 0, using the same image, CONFIG=next, GPU utilization 0.78, and 17 matching workloads/seeds. Per-phase cache salts separate warmups, scenarios, and runs; requests within shared-prefix phases retain intentional prefix reuse. Exact output lengths were validated by the harness.

The [serving table](shared_vs_next_tasks_1.serving.md) contains all task rates, TTFT, ITL, and measured-phase MTP acceptance. The [JSON analysis](shared_vs_next_tasks_1.analysis.json) contains raw values, differences, MTP counter deltas, and source SHA256s.

## Serving and MTP limits

The seven throughput scenarios improve 0.2–5.2% in this pair. Single-request prefill median TTFT improves 0.8–14.3% (8k: 191.6→164.3 ms; 16k: 384.5→329.5 ms; 32k: 866.3→756.6 ms). Decode output throughput changes +10.3%, −3.0%, +2.2%, −1.7% at c=1/4/16/64. These are one run per variant, not confidence intervals or an adoption decision.

Serving commands omit temperature and request sampling seed. Both startup logs report model defaults temperature=1.0, top_k=20, top_p=0.95. Dataset seeds fix prompts, not generated token paths. For decode c1, MTP acceptance rises 44.1→53.5% while median ITL is 6.52→6.56 ms; c4 acceptance falls 64.1→61.1% with ITL 8.21→8.32 ms. Thus these rates do not isolate kernel execution. The earlier short-pair ≈6% decode loss does not reproduce uniformly; explicit-greedy repeated A/B is still needed. No throughput change is attributed to the load-time 1-ULP alpha differences.

## Quality: unresolved regressions

Quality probes explicitly request temperature=0. Prompt-logprob requests skip prefix-cache reads in the installed vLLM, so the two repeated dumps test cold prefill. They are not deterministic even within one server. Neither server has an identical greedy continuation on any of its four repeated prompts (0/4 each). That variability limits attribution; it does not establish quality equivalence.

| prompt tokens | next NLL, two cold calls | shared NLL, two cold calls | change in two-call mean |
|---:|---:|---:|---:|
| 788 | 0.23386 / 0.24812 | 0.21131 / 0.23514 | -7.4% |
| 669 | 0.45699 / 0.41524 | 0.41823 / 0.41250 | -4.8% |
| 3443 | 0.38302 / 0.33306 | 0.37925 / 0.44676 | +15.4% |
| 7604 | 0.57373 / 0.57051 | 0.58248 / 0.57158 | +0.9% |

The 3,443-token mean NLL worsens by 0.05497 (+15.4%); within-run repeat shifts are −0.04996 for next and +0.06751 for shared. At 7,604 tokens the mean increases by 0.00491 (+0.9%). Mean absolute logprob differences between variants (0.121–0.209) overlap the within-run range (0.107–0.216), including the sub-1024 CUTLASS-only prompts. Two repeats cannot rule out regression.

Long-document NLL has only one sample per document/length:

| document | scored tokens | next NLL | shared NLL | change |
|---|---:|---:|---:|---:|
| licenses | 6349 | 0.10042 | 0.10627 | +5.8% |
| licenses | 12918 | 0.08355 | 0.08383 | +0.3% |
| py_stdlib | 6855 | 0.53204 | 0.52523 | -1.3% |
| py_stdlib | 13128 | 0.47865 | 0.47746 | -0.2% |
| py_email | 7689 | 0.79454 | 0.79683 | +0.3% |
| py_email | 15224 | 0.78229 | 0.78617 | +0.5% |

Flag the 6,349-token licenses regression: 0.10042→0.10627 NLL (+5.8%), with last-half NLL 0.13368→0.14683 (+9.8%). Both needle probes are 20/20. Checkpoint hot-versus-cold mean TV over top-20 tokens worsens 0.1493→0.1710, maximum 0.4203→0.4629, and top-1 agreement falls 11/12→9/12 at the same 6,709 prefix hits per prompt. This is within-server hot/cold consistency, not direct candidate-versus-baseline TV or full-vocabulary TV. It also needs repetition.

The candidate remains optional. Exact-alpha and scalar experiments have separate artifacts; these serving/quality numbers do not validate either later variant. Agentic evaluation remains reserved for the combined configuration.
