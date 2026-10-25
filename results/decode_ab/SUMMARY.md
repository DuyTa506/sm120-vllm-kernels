# Kernel page size vs decode speed

Question: the optimized config uses a 64-token FlashInfer kernel page (`VLLM_FI_KERNEL_PAGE64=1`) only so the hybrid
manager block becomes a multiple of 64 (2176), which patch 02's internal GDN checkpoint needs. Does the 64-token page
slow decode, and can the cache block (2176) be decoupled from the kernel page (16)?

Patch 04 (`VLLM_FI_KERNEL_PAGES=16`) overrides the FlashInfer backend's supported kernel page list, so
`--block-size 2176` runs on 16-token kernel pages.

| run | manager block | kernel page | other flags | KV tokens |
|---|---:|---:|---|---:|
| A | 2128 | 16 | tuned | 3,538,944 |
| B | 2176 | 64 | tuned + page64 | 3,508,420 |
| C | 2176 | 16 | tuned + FP8 prefill + GDN internal ckpt + patch 04 | 3,369,930 |

`bench/bench_decode.sh` (random 128 in / 512 out, ignore_eos, MTP-2), two runs per config with seeds 600 and 700.
Files: `{A,B,C}_*.rep{1,2}.txt`, `C_block2176_page16.prefill.txt`.

## Decode, output tok/s (run 1 / run 2)

| c | A | B | C |
|---:|---|---|---|
| 1 | 350 / 330 | 324 / 295 | 333 / 331 |
| 4 | 943 / 813 | 845 / 879 | 815 / 831 |
| 16 | 1899 / 1854 | 1902 / 1887 | 1870 / 1980 |
| 64 | **3677 / 3673** | **3532 / 3524** | **3719 / 3575** |

Median ITL (ms) at c=64: A 28.79 / 28.40, B 29.02 / 28.79, C 29.02 / 28.96.

- At c=1 and c=4 run-to-run spread (up to 14%) exceeds any config difference; speculative acceptance on random
  prompts dominates. Not usable for A/B.
- At c=64 B is 4.0% below A in both runs. C averages 3647 (−0.8% vs A, within its own 4% spread).
- MTP acceptance over the whole run is the same for B and C (accepted / draft tokens: B 59.0%, C 58.7%), and median
  ITL barely moves, so B's loss is not acceptance and not the median step. It sits in the slower steps (mean TPOT
  +5%), most likely the mixed prefill + decode steps when new requests join.

## C keeps the prefill gains

Single-request median TTFT (ms), `bench/bench_prefill.sh`:

| prompt | tuned | optimized (page 64) | C (page 16) |
|---:|---:|---:|---:|
| 2048 | 85.6 | 88.1 | 72.7 |
| 4096 | 110.3 | 118.3 | 106.1 |
| 5000 | 157.4 | 138.9 | 131.1 |
| 8192 | 230.2 | 201.6 | 199.2 |
| 16384 | 470.3 | 398.3 | 388.4 |
| 32768 | 1100.4 | 885.0 | 892.2 |

Prefix-cache correctness (`eval/ckpt_dist.py abC`): 6709 cached tokens per prompt, mean TV(top20) 0.113, max 0.268,
top-1 agreement 11/12 (stock vLLM 0.12–0.13; broken checkpoint 0.79).

## Verdict

Decoupling works: block 2176 for the cache (one-pass GDN checkpoint), 16-token kernel pages for attention. It recovers
the ~4% high-concurrency decode loss of page 64 without giving up any prefill gain. The KV capacity drop of the
optimized config (3.54M → 3.37M) comes from patches 01/02, not from the page size (B keeps 3.51M).
