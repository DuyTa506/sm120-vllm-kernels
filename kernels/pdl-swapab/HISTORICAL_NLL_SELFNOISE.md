# Baseline repeated-NLL evidence

This compares repeats within each baseline, not OFF/ON differences. It is descriptive evidence of existing variability, not a quality acceptance threshold or causal explanation.

| Baseline artifact under `results/qwen_sequential/` | Rounds | logprobs-2 NLL sequence | Range (nat/token) |
|---|---:|---|---:|
| `next_diagnostic_1/repeated_nll.json` | 5 | 0.361797 / 0.344979 / 0.385570 / 0.367218 / 0.357493 | 0.040592 |
| `next_matched_3/repeated_nll.json` | 3 | 0.384609 / 0.383348 / 0.360658 | 0.023951 |
| `scalar_matched_3/repeated_nll.json` | 3 | 0.394169 / 0.326924 / 0.363132 | 0.067245 |
| `scalar_focus_1/repeated_nll.json` | 5 | 0.370921 / 0.438515 / 0.373984 / 0.347137 / 0.346903 | 0.091613 |

All these logprobs-2 prompts have 3,443 tokens and identical scored-token hashes across repeats. `scalar_focus_1/config.txt` records config=next, variant=scalar, FI control enabled; `next_diagnostic_1` is next, with no FI-control field. These historical compiler/runtime settings are not asserted identical to the current task-order run.

Task-order OFF a1 has 4,096 tokens and NLL 0.240264 / 0.219173 / 0.199898 / 0.191320, range 0.048944. Historical long-corpus six-prompt ranges are smaller: scalar_focus_1 0.006561–0.013496; short logprobs probes vary more (0.009978–0.091613). Current monotonic decline is not universal historically: scalar_focus_1 logprobs-2 rises first then falls, while next_diagnostic_1 oscillates.

Historical scalar_focus_1 logprobs-2 repeat-vs-first per-token RMS differences are 0.727–0.818, with 589–649 of 3,442 positions differing by >0.1 nat and maximum differences 10.58–14.55 nat. Task-order has RMS 0.829–0.877, 455–466 of 4,095 positions >0.1 nat. Therefore historical variability also includes substantial localized changes, not just averaging noise. Different prompt content prevents treating these distributions as interchangeable.

## Cache methodology

- `eval/repeated_nll.py:89` schedules unique cache salts per request and rotates ten interleaved prompts each round. `run()` sends the salt separately from prompt text, uses temperature=0 / max_tokens=1 / prompt_logprobs=0, and never calls reset_prefix_cache. It validates the same prompt tokenization per repeat.
- `kernels/moe-task-order-serving/tasks.py:15` waits for successful reset_prefix_cache before every request, sends identical token IDs four times consecutively, no cache_salt, and ignore_eos=True.
- Both run in one live server. “Cold” here denotes prefix-isolated requests, not independent server startups. Salt changes the cache namespace, not the prompt text; successful reset confirms the endpoint accepted the cache reset, not that every model-state/allocator buffer was cleared.
- Reset versus salted isolation and request ordering could change state/cache allocation histories. There is no matched experiment here to establish that either explains the numerical drift. Baseline self-variation alone cannot make a patch quality-equivalent.

Full per-prompt/per-token aggregate evidence with input file hashes is in `results/historical_cold_nll_selfnoise.json`. Source task-order evidence is `kernels/moe-task-order-serving/results/task_order_serving_a1/tasks.json` and `root_nll_repeat_diagnosis.json`. No frozen PDL sources were changed and no GPU run was launched.
