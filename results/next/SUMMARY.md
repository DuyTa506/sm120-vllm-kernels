# next config: optimized + patches 03, 04, 05

`CONFIG=next scripts/serve.sh` (after `scripts/build_patched.sh`):

| patch | flag | what |
|---|---|---|
| 03 | `VLLM_SM120_FP8_PREFILL_NOGLUE=1` | FP8 FMHA prefill in one kernel: strided K/V straight from vLLM's cache, Q quantized in-kernel (`kernels/fp8-attn-glue`) |
| 04 | `VLLM_FI_KERNEL_PAGES=16` + `--block-size 2176` | 64-aligned cache block for the GDN checkpoint, 16-token attention kernel pages (`results/decode_ab`) |
| 05 | `VLLM_GDN_FUSED_CONV=1` | GDN causal conv1d + post-conv prep in one Triton kernel (`kernels/gdn-conv-fusion`) |

`build/` made from patches 01-05 is byte-identical to the build these numbers were measured on.

## Prefill GPU span (nsys, same session, `results/nsys/span_next_vs_optimized.txt`)

| prompt | optimized | next | change |
|---:|---:|---:|---:|
| 8192 | 136.8 ms, 750 kernels + 44 graphs, 91% busy | **125.1 ms**, 678 kernels + 44 graphs, 94% busy | −8.6% |
| 32768 | 573.9 ms | **551.4 ms** | −3.9% |

## TTFT, single request, median (`ttft_next_p16.txt`)

| prompt | tuned | optimized | next |
|---:|---:|---:|---:|
| 2048 | 85.6 | 88.1 | **72.4** |
| 4096 | 110.3 | 118.3 | **109.3** |
| 5000 | 157.4 | 138.9 | **119.0** |
| 8192 | 230.2 | 201.6 | **189.8** |
| 16384 | 470.3 | 398.3 | **383.5** |
| 32768 | 1100.4 | 885.0 | **856.3** |

## Decode (`decode_next_p16.rep{1,2}.txt`, same seeds as `results/decode_ab`)

c=64: 3561 / 3630 tok/s, median ITL 28.91 / 28.85 ms. For comparison: tuned 3677 / 3673, page-64 (as in optimized)
3532 / 3524, optimized with page 16 3719 / 3575. Decode is unchanged by 03 and 05 (prefill-only); 04 removes the
page-64 loss. c=1-16 are within the MTP acceptance spread.

## Correctness

- Prefix-cache checkpoint (`eval/ckpt_dist.py next_p16`): 6709 cached tokens per prompt, mean TV(top20) 0.118,
  max 0.292, top-1 agreement 10/12 (stock 0.12-0.13).
- Earlier smoke run of the same patches on page 64: NLL licenses within noise of optimized, needles 20/20, no
  server errors (`nll_lic.json`, `long_eval.json`).
- Patch 03 output is bitwise identical to patch 01's path; patch 05 is checked against the unfused kernels in
  `kernels/gdn-conv-fusion/test_correctness.py`.

## Throughput sweep (`results/throughput/next.tsv`, `vllm bench serve`, random prompts, fixed seeds, one run per config)

| scenario | optimized total tok/s | next | change | median TTFT optimized / next (ms) |
|---|---:|---:|---:|---:|
| long-doc 32k, c=2 | 30923 | 30633 | -0.9% | 1193 / 1020 |
| prefill-heavy 8k, c=8 | 40748 | 40880 | +0.3% | 372 / 448 |
| rag 4k, c=16 | 23485 | 23394 | -0.4% | 201 / 212 |
| shared-prefix, c=16 | 26780 | 26427 | -1.3% | 261 / 238 |
| mixed-len 4k, c=32 | 18372 | 19677 | +7.1% | 251 / 217 |
| chat 1k, c=32 | 7928 | 8166 | +3.0% | 167 / 152 |
| decode-heavy, c=64 | 4498 | 4579 | +1.8% | 173 / 182 |

Most rows are inside single-run noise. The gain of `next` is prefill latency (span and TTFT above), not mixed-load
throughput. Long-doc median TPOT is worse (6.0 -> 7.5 ms); not yet re-measured.

The agentic replay of `next` was not run: agentic runs are reserved for the final combination of kernels that win
their individual A/B.
