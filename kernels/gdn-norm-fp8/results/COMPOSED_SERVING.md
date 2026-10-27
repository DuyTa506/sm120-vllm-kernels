# Scalar + GDN full serving comparison: mixed, not selected

`scalar_gdn_matched_3` completed successfully on Qwen3.6-35B-A3B-NVFP4-Fast,
vLLM 0.32.0 and the RTX PRO 6000 Blackwell. The composition improves long-doc
throughput and 32k cold prefill versus the scalar parent, but regresses 8k cold
prefill and greedy decode at c=64. GDN remains default-off and **is not selected
for the current final stack** after the completed focused repeat below. Scalar
without this additional fusion completes final agentic evaluation. This
is a scoped configuration choice, not a general rejection of GDN fusion. No
per-patch agentic run was made.

## Completed run and controls

The [raw run](../../../results/qwen_sequential/scalar_gdn_matched_3) has exit 0,
all 17 serving cases, four explicit-greedy decode cases (512 measured requests),
30 cold NLL calls (ten prompts × three rounds), and 20/20 needle responses.
All 30 expected GDN adapters report ready; the captured 29 source files match
their SHA256 manifest and cleanup source verification passes. The scalar
parent has 23 captured files and `next` has 21; their captured hashes also match.

All three comparisons use GPU utilization 0.78, MTP2, TP1 and the same fixed
FlashInfer reference. Startup and shutdown each verify all 132 entries, with
effective-entry hash
`721ed789107ca7cd8d54b4ce8e6345a9e84e8882beca5914a4a2085598f1b063`.
There is no serving lookup observer. The serving matrix retains the model's
sampling settings; the separate greedy cases set temperature 0. Both complete
parents and the GDN run use the same `individual` suite before those diagnostics.

Inductor is observed, not pinned. Relative to scalar, GDN adds three and removes
four tuning records, with four common records changed. Relative to `next`, two
common records change. These are saved tuning-file observations; they do not
prove which candidate ran at each launch or establish a compiler cause.
The GDN composition also adds its opaque split op while preserving the existing
19 splits and `fuse_rope_kvcache_cat_mla=False`.

Logged startup allocation differs even at the same memory budget:

| configuration | available KV GiB | KV token capacity | final CUDA graph pool GiB |
|---|---:|---:|---:|
| next | 44.61 | 3,369,930 | 4.90 |
| scalar | 45.50 | 3,437,190 | 3.45 |
| scalar + GDN | 45.62 | 3,445,813 | 3.33 |

These rounded log values describe startup capacity, not a peak-memory benchmark
or an isolated saving caused by the GDN kernel.

## Cold prefill

Each case has eight measured requests at concurrency 1 and one output token.
TTFT p50 in milliseconds; negative percentage means lower latency:

| input tokens | next | scalar | scalar + GDN | GDN vs scalar | stack vs next |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 70.53 | 68.19 | 63.68 | -6.6% | -9.7% |
| 4,096 | 108.95 | 100.54 | 100.06 | -0.5% | -8.2% |
| 5,000 | 122.18 | 115.44 | 111.49 | -3.4% | -8.8% |
| 8,192 | 201.98 | 164.08 | 176.22 | +7.4% | -12.8% |
| 16,384 | 379.35 | 343.63 | 348.44 | +1.4% | -8.1% |
| 32,768 | 873.31 | 811.50 | 750.03 | -7.6% | -14.1% |

The isolated layer-18 8k chain improves 391→287 µs, while this full-server 8k
case worsens versus scalar. A microbenchmark win therefore does not establish
an end-to-end win for every request shape.

## Serving and decode

Incremental total throughput versus scalar, across the seven existing workloads:

| workload | scalar tok/s | scalar + GDN tok/s | change |
|---|---:|---:|---:|
| long-doc | 32,168.5 | 35,097.4 | +9.1% |
| prefill-heavy | 43,102.3 | 43,824.4 | +1.7% |
| rag | 23,545.2 | 24,028.1 | +2.1% |
| shared-prefix | 27,767.8 | 28,388.3 | +2.2% |
| mixed-len | 19,374.6 | 19,294.5 | -0.4% |
| chat | 8,138.9 | 8,591.7 | +5.6% |
| decode-heavy | 4,567.7 | 4,432.6 | -3.0% |

Across these workloads, the complete stack versus `next` changes throughput
by +12.2%, +5.8%, +5.5%, +7.6%, +2.8%, +4.1% and -3.9%, respectively.
MTP acceptance and generated paths also vary; the full throughput change is
not attributed entirely to norm/quant fusion.

The four sampled-decode output rates all decline versus scalar, by 3.4–7.8%.
The separate explicit-greedy result is mixed:

| concurrency | output rate change vs scalar | output rate change vs next |
|---:|---:|---:|
| 1 | +2.17% | +4.30% |
| 4 | -3.74% | -0.20% |
| 16 | +5.00% | +1.11% |
| 64 | -8.71% | -8.17% |

At greedy c=64, scalar→GDN output rate is 5,015.37→4,578.64 tok/s and median
TPOT is 11.766→13.284 ms. Median ITL is slightly lower (27.462→27.342 ms),
while MTP acceptance is almost unchanged (85.56%→85.73%). Median ITL or
acceptance alone therefore does not explain the lower throughput. These
observations do not identify its cause.

## Quality diagnostics and remaining selection

All three runs pass 20/20 needles, which is a narrow functional check. In the
three-round cold NLL comparison, nine of ten means rise when adding GDN to
scalar. All observed incremental ranges overlap, but overlap is neither a
quality pass nor evidence that every difference is noise.

The clearest retained stack-versus-`next` flag is `licenses-30000`:

| configuration | mean NLL | observed min–max |
|---|---:|---:|
| next | 0.094275 | 0.089399–0.097411 |
| scalar | 0.096490 | 0.093787–0.101402 |
| scalar + GDN | 0.104989 | 0.100266–0.111138 |

The stack's +0.010715 mean delta versus `next` has nonoverlapping observed
ranges. Versus scalar the delta is +0.008500 and the ranges overlap slightly.
Conversely, `py_email-30000` improves from scalar's 0.802070 to 0.797925.
This does not erase the earlier scalar-versus-exact `py_email-30000` flag.
These ten prompts and three repeats are diagnostic only; observed ranges are
not confidence bounds or an acceptance threshold. Kernel/state exactness does
not establish full-model quality equivalence.

## Completed focused repeat and selection

`gdn_focus_1` then `scalar_focus_1` reverses the original execution order.
Both exit 0 with the unchanged six cold-prefill cases, four greedy cases and
50 cold NLL calls (five rounds per prompt). The focused suite omits the seven
throughput cases, legacy quality probes and agentic. Within this pair the prior
suite matches; its shorter history differs from the original full pair.

The GDN/scalar captured sources match all 29/23 manifest hashes, all 30 GDN
adapters are ready, and both 132-entry FI startup/shutdown gates match the same
reference as above. KV and graph allocations reproduce the values above.
Inductor remains observed, not pinned, with four common records changed, three
added and four removed relative to scalar.

| cold prefill tokens | scalar TTFT ms | scalar + GDN TTFT ms | focused change | original full change |
|---:|---:|---:|---:|---:|
| 2,048 | 73.51 | 63.95 | -13.00% | -6.62% |
| 4,096 | 94.10 | 93.39 | -0.75% | -0.48% |
| 5,000 | 112.28 | 108.12 | -3.71% | -3.43% |
| 8,192 | 173.44 | 160.76 | -7.31% | +7.40% |
| 16,384 | 354.65 | 351.99 | -0.75% | +1.40% |
| 32,768 | 793.75 | 778.03 | -1.98% | -7.57% |

| greedy concurrency | scalar output tok/s | scalar + GDN output tok/s | focused change | original full change |
|---:|---:|---:|---:|---:|
| 1 | 433.54 | 440.39 | +1.58% | +2.17% |
| 4 | 1,251.20 | 1,198.95 | -4.18% | -3.74% |
| 16 | 2,643.82 | 2,440.33 | -7.70% | +5.00% |
| 64 | 5,078.89 | 4,873.45 | -4.04% | -8.71% |

The 8k prefill direction changes, 16k stays near neutral and the 32k benefit
shrinks. Greedy c=4/c=64 losses recur, while the c=16 result changes sign.
This is enough to prefer scalar for the current shared final configuration;
it does not establish that GDN is slower for every prefill or serving workload.

Eight of ten focused NLL means rise with GDN, with overlapping five-repeat
ranges for all ten. For `licenses-30000`, scalar is 0.100256
[0.093799, 0.107295] and GDN is 0.100381 [0.092199, 0.107727], a +0.000125
delta. The earlier large incremental licenses delta does not reproduce at
that size. This pair does not include `next`, so the original stack-versus-next
three-repeat flag remains historical evidence rather than being retested or
resolved. Neither overlap nor the smaller delta establishes quality equivalence.

The selected final stack is root `next` patches 01–05 plus MoE patches 03–06,
without this additional GDN norm/FP8 overlay. The [selection record](../../../results/qwen_sequential/final_stack_selection.md)
contains the exact composition and commands. Only that selected complete stack
entered final agentic against `next` (`next_final_agentic_1` and
`stack_final_agentic_1`). The [completed result](../../../results/qwen_sequential/final_agentic_analysis.md)
is modest and mixed: output-rate changes -3.60% to +3.92%, with no robust/general
speedup established from one pair. It does not measure GDN agentic performance;
GDN remains opt-in for further scoped work and default `next` is unchanged.

The [private-state probe](STATE_PROBE_GPU.md) remains scoped to layer 18 and
actual eager long-prefill, mixed speculative-decode/prefill and pure-decode
metadata. Its six ordinary/padded variants match Q/S and private state bytes;
checkpoint-bearing and mixed non-speculative metadata were not observed.
The [patch composition proof](../PATCH_COMPOSITION.md) checks source identity
in either application order. Neither proof settles these serving tradeoffs.

## Comparison artifacts

- Incremental GDN versus scalar: [serving](../../../results/qwen_sequential/gdn_vs_scalar_matched_3.serving.md), [greedy decode](../../../results/qwen_sequential/gdn_vs_scalar_matched_3.greedy.md), [cold NLL](../../../results/qwen_sequential/gdn_vs_scalar_matched_3.nll.txt).
- Complete stack versus `next`: [serving](../../../results/qwen_sequential/stack_vs_next_matched_3.serving.md), [greedy decode](../../../results/qwen_sequential/stack_vs_next_matched_3.greedy.md), [cold NLL](../../../results/qwen_sequential/stack_vs_next_matched_3.nll.txt).

- Reversed focused GDN versus scalar: [prefill](../../../results/qwen_sequential/gdn_vs_scalar_focus_1.prefill.md), [greedy decode](../../../results/qwen_sequential/gdn_vs_scalar_focus_1.greedy.md), [five-round cold NLL](../../../results/qwen_sequential/gdn_vs_scalar_focus_1.nll.txt).

Each pair is a configuration comparison without a timing confidence interval.
Their suite histories differ, so they are not pooled as interchangeable timing
samples. The archived artifacts retain actual counts, settings, source identity,
tuning observations and MTP counters for review.
