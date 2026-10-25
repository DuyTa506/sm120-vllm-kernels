# Agentic benchmark results

Token-exact replay of tool-calling sessions recorded live on the production config (`bench/agentic_bench.py`).
short: 8 coding sessions, 16 turns, 10-30k tokens. long: 8 whole-package reviews, up to 120 turns, up to ~126k tokens.
More agents than sessions replay copies under separate cache salts. TTFT is for turns after the first.

- live short / production: 128 turns, bad tool-call JSON 0, tool errors 1, truncated 0, context full 0
- live short / tuned: 128 turns, bad tool-call JSON 0, tool errors 4, truncated 0, context full 0
- live short / optimized: 125 turns, bad tool-call JSON 0, tool errors 11, truncated 0, context full 0
- live long / production: 770 turns, bad tool-call JSON 0, tool errors 131, truncated 0, context full 3
- live long / tuned: 577 turns, bad tool-call JSON 0, tool errors 11, truncated 0, context full 5
- live long / optimized: 658 turns, bad tool-call JSON 0, tool errors 52, truncated 0, context full 5

| set | agents | config | turns/s | out tok/s | TTFT p50 ms | TTFT p90 ms | prefix hit | session s |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| short | 1 | production | 1.21 | 309 | 205 | 363 | 69.4% | 13 |
| short | 1 | tuned | 1.33 | 341 | 142 | 195 | 69.4% | 12 |
| short | 1 | optimized | 1.41 | 362 | 111 | 144 | 68.8% | 11 |
| short | 4 | production | 2.39 | 613 | 231 | 411 | 69.4% | 25 |
| short | 4 | tuned | 3.06 | 782 | 177 | 271 | 69.4% | 20 |
| short | 4 | optimized | 2.74 | 701 | 131 | 200 | 68.8% | 22 |
| short | 8 | production | 2.63 | 672 | 239 | 439 | 69.4% | 33 |
| short | 8 | tuned | 3.35 | 857 | 176 | 299 | 69.4% | 26 |
| short | 8 | optimized | 3.66 | 936 | 143 | 227 | 68.8% | 23 |
| short | 16 | production | 3.35 | 858 | 412 | 658 | 69.4% | 53 |
| short | 16 | tuned | 4.47 | 1145 | 231 | 543 | 69.4% | 40 |
| short | 16 | optimized | 4.86 | 1245 | 194 | 377 | 68.8% | 36 |
| short | 32 | production | 4.37 | 1119 | 443 | 785 | 69.4% | 85 |
| short | 32 | tuned | 5.44 | 1392 | 322 | 719 | 69.4% | 66 |
| short | 32 | optimized | 6.03 | 1542 | 271 | 626 | 68.8% | 60 |
| long | 1 | production | 1.23 | 185 | 433 | 624 | 94.7% | 78 |
| long | 1 | tuned | 1.27 | 191 | 415 | 614 | 94.7% | 76 |
| long | 1 | optimized | 1.59 | 239 | 279 | 373 | 94.5% | 60 |
| long | 8 | production | 1.94 | 291 | 590 | 1354 | 94.7% | 249 |
| long | 8 | tuned | 1.99 | 298 | 562 | 1529 | 94.7% | 241 |
| long | 8 | optimized | 2.81 | 422 | 373 | 826 | 94.5% | 167 |
| long | 32 | production | 2.32 | 349 | 1584 | 3785 | 94.2% | 851 |
| long | 32 | tuned | 2.36 | 354 | 1498 | 3266 | 94.2% | 842 |
| long | 32 | optimized | 3.42 | 513 | 997 | 1992 | 94.0% | 577 |

## Caching experiments (interim, from `kernels/cache-granularity` and `kernels/session-cache`)

A/B on one server (production config), same recorded turns. "prefilled" = tokens prefilled per follow-up turn.

| experiment | workload | stock: prefilled / TTFT p50 | finer: prefilled / TTFT p50 |
|---|---|---|---|
| `--prefix-match-unit 16` | long, 4 sessions, 1 agent | 4,759 / 486 ms | 1,765 / 330 ms |
| `--prefix-match-unit 16` | long, 4 sessions, 8 agents | 4,759 / 1,040 ms (p90 2,890) | 1,765 / 827 ms (p90 2,448) |
| session state cache | long, 4 sessions from turn 20, 1 agent | 5,460 / 532 ms | 2,415 / 351 ms |
| session state cache | short, 4 sessions, history resent exactly | 4,172 / 219 ms | 1,213 / 96 ms |
| session state cache | short, history as recorded (rewritten), stage 1 run | 4,460 / 203 ms | 4,460 / 355 ms (regression, being checked) |

Truly new tokens per long-session turn: ~1,740. Prefix-cache check with `--prefix-match-unit 16`: TV 0.122
(stock 0.12-0.13).

## next config, smoke replay (page 64, before the page-16 change)

| set | agents | turns/s | out tok/s | TTFT p50 | TTFT p90 |
|---|---:|---:|---:|---:|---:|
| short | 1 | 1.41 | 362 | 111 ms | 144 ms |
| short | 8 | 3.08 | 789 | 136 ms | 224 ms |

TTFT matches or beats optimized; turns/s at 8 agents is below optimized (3.66), within the decode run-to-run
spread seen elsewhere (MTP acceptance on replayed turns). A full replay on next with page 16 is queued.
