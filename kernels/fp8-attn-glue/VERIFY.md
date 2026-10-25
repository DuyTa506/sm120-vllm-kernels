# VERIFY: fp8-attn-glue (patch 03, `VLLM_SM120_FP8_PREFILL_NOGLUE`)

Independent verifier notes, 2026-10-25. I did not modify the author's code. I added these
files: `verify_correctness.py`, `verify_bench.py`, `verify_host_overhead.py` and
`verify_nsys_attn.py`. Raw logs and the in-server launcher are in the session scratchpad.
The relevant numbers are copied below.

## Verdict: CONFIRMED, with the size of the gain slightly overstated

* **Bitwise identical output: confirmed.** I re-ran the author's `correctness.py` (ALL OK)
  and added cases it did not cover. All are bitwise equal to the patch-01 path.
* **About 90 us per call at 8k: reproduced** with the author's method (88.0 us; 16k 272 us,
  32k 662 us, 32k chunk 322 us, varlen 93 us).
* **About 1 ms per 8k prefill: roughly holds; the realistic figure is 0.75 to 0.95 ms.**
  In the real server (nsys, 8 requests per side, two runs in opposite order), GPU busy time
  per 8k request fell by 0.96 ms and 0.78 ms. When Q is warm in L2, as it is right after
  RoPE in the model, the microbenchmark saving is 65 us per call (0.72 ms per request), not
  90 us.
* **GPU span and TTFT end to end: not resolvable.** Whichever server launches second in a
  lock hold runs 3 to 4.5 ms slower per request. That swamps the gain (details below).

## 1. Correctness (re-run under the GPU lock)

Author's `correctness.py`: all bitwise checks max_abs 0. FP32-reference error is identical
on both paths (1.9e-3 / 6.3e-5 / 2.3e-3 / 2.2e-3). This matches their results file.

`verify_correctness.py` covers cases the author did not test. Each is checked bitwise
against `common.run_baseline`, which replicates the patch-01 body, and each passed with 0
mismatched elements:

| case | why it matters | result |
|---|---|---|
| a. 5.2 GB pools (80k blocks compact, 40k blocks with x2 padded block stride), pages at high block ids, max byte offset 4.88 GiB, HND and NHD | the new TMA descriptor spans the **whole** per-layer cache. The baseline only ever saw a small gathered copy, and the author's pools were 256 MB. In-server runs only touch low block ids. | OK (4/4) |
| b. q_scale 0.7071 / 1.5 / 0.125 on a 3-sequence varlen batch with prefix, amplitude 40 (heavy saturation) | the author tested q_scale != 1 only on a 16-row Q | OK (3/3) |
| c. mixed batch: prefill Q/out are slices after 3 / 37 / 192 decode rows | vLLM passes `query[num_decode_tokens:]` | OK, decode rows untouched |
| d. 5 repeated runs at 8k | determinism | 0 mismatches |
| e. CUDA graph capture + replay of the new launcher | capture safety (prefill attention is eager in vLLM) | OK |
| f. block-table entries past each request's last page set to 2^30 | the absolute table must never be read out of range | OK |

`verify_host_overhead.py` calls the **real patched** `_sm120_fp8_paged_prefill`, with the
patch-03 `flashinfer.py` and `sm120_fp8_fmha_glue.py` bind-mounted, once with `_noglue=False`
(the patch-01 code) and once with `_noglue=True`. The output is bitwise equal at 512, 8k and
varlen.

The author's in-server log (`results/e2e_verify.txt`) checks out: 132 calls with 0
mismatched elements across 6 distinct shapes, 11 calls per forward.

## 2. Timing (CUDA events, eager, no CUDA graphs, interleaved A/D)

Author's `bench.py --reps 50`, re-run (us per call):

| case | A patch 01 | D new | A-D (mine) | A-D (author) |
|---|---|---|---|---|
| 8k | 991.2 | 903.2 | **88.0** | 90.1 |
| 16k | 3886.1 | 3614.2 | 271.9 | 272.4 |
| 32k | 14950.4 | 14287.9 | 662.5 | 678.9 |
| 32k 2nd chunk | 11051.0 | 10729.0 | 322.0 | 324.1 |
| varlen (4 seqs) | 508.9 | 415.7 | 93.2 | 92.2 |

The absolute FMHA times in my run are about 3.5% slower than the author's (892 vs 862 us at
8k, about 615 TFLOPS rather than 640). That is the GPU's clock or thermal state; the A-D
differences reproduce. I did not re-run the tile sweep.

`verify_bench.py` (60 reps, interleaved, median [IQR]) tests how sensitive the saving is to
L2 state:

| case | flush (author's) | Q hot in L2 (flush, then a copy kernel writes Q) | warm (no flush) | back-to-back x20 |
|---|---|---|---|---|
| 8k | 90.1 | **65.5** | 86.3 | 74.5 |
| varlen | 91.1 | 73.7 | 81.3 | 85.2 |
| 2k | 24.1 | 18.4 | 50.7 | 27.0 |
| 512 | 22.5 | 20.5 | 52.3 | 41.7 |
| 16k | 263.2 | 285.2 | 246.5 | 261.4 |

In the model, Q comes straight out of q-norm/RoPE and is partly L2-resident. The
"Q hot" column (65 us at 8k, so 0.72 ms per 8k request) is therefore the more realistic
microbenchmark number. The author's 0.99 ms is the cold-L2 upper end.

Host cost per call of the real function (`verify_host_overhead.py`): patch 01 takes 40 to
77 us of CPU per call and patch 03 takes 14 to 22 us. The new path does **not** add CPU
launch overhead; it removes some.

## 3. In-server A/B (nsys, real weights, "optimized" flags + MTP-2 + patch 02)

The launcher is `scratchpad/verify_e2e_nsys.sh`. It runs the same mounts and flags as the
author's `e2e.sh`, under `nsys launch -t cuda,nvtx`. For each server it sends 3 warmups,
then records 8 single 8k-token prefills (c=1, output 1 token, the same seed on both sides,
no prefix-cache hits). I ran it twice, in opposite order.

| run (order) | NG=0 busy / span (ms per request) | NG=1 busy / span | busy saved | span delta |
|---|---|---|---|---|
| run 1 (0 then 1) | 127.07 / 135.76 | 126.11 / 138.60 | **0.96** | -2.84 (NG=1 slower) |
| run 2 (1 then 0) | 126.98 / 139.11 | 126.20 / 135.65 | **0.78** | +3.46 (NG=1 faster) |
| mean of both orders | 137.4 span | 137.1 span | 0.87 | +0.3 (noise ±3.5) |

Kernel-level breakdown (median per request, `verify_nsys_attn.py`):

* Removed: clamp 574 us (12 to 1 calls), the index_select gathers 214 us (22 calls) and the
  FP8 convert 208 us (11 calls), for about 996 us.
* FMHA: 8382-8400 us becomes 8561-8597 us, so +180 to 215 us (about 16 to 20 us per call).
  That is the in-kernel BF16 load and convert. It costs more than the author's
  microbenchmark figure of about 10 us.
* Eager attention window between piecewise graphs (run 1): 10.05 ms becomes 9.15 ms per
  request, **-0.90 ms**.

TTFT (`vllm bench serve`, 8 prompts): run 1 gives 193.1 (NG=0) and 198.6 (NG=1); run 2
gives 193.6 (NG=1) and 199.4 (NG=0). Mean TTFT tracks launch order, not the flag. The extra
idle in the second server is spread over non-attention transitions (`_apply_write_kernel`,
step-preparation kernels, `_prepare_rope_positions`), so it is a host or launch-order
effect, not this patch. **Note for every e2e A/B in this project:** the author's `e2e.sh`
always runs NOGLUE=0 first. That order penalises the NOGLUE=1 server by about 3 ms, which
alone explains their "unchanged within noise" TTFT. Always alternate the order.

## 4. Things checked that are fine

* Same work on both sides: the bench baseline is a faithful copy of the patch-01 body
  (diffed against `build/`). It uses the same compiled FlashInfer kernel: q128/kv128 and
  `_use_balanced_scheduler(True)`, which is True unless `PRIMS_FMHA_DISABLE_BALANCED_SCHEDULING=1`.
  Metadata is built outside the timed region on both sides, as in vLLM, where it is cached
  per step.
* `patch/a/.../flashinfer.py` is identical to `build/vllm/.../flashinfer.py`.
  `sm120_fp8_fmha_glue.py` equals `glue_fmha.py` except for the docstring.
* Decode and spec-decode verification (q_len 3) never take this path (`MIN_Q`=256), so
  there is no decode effect, as claimed. Mixed decode+prefill batches are covered by
  case c.
* Memory: per layer the patch removes the transient K/V copies (2x4 MB at 8k) and the
  32 MB FP8 Q. It adds nothing.
* The kernel compiles during the profile run (the 64x256 warmup has `max_q`=256, which is
  >= MIN_Q), not on the first user request. It is disk-cached.

## 5. Integration risks / open items

1. **The q_scale is baked in at compile time** (`functools.cache` key). Every distinct
   `layer._q_scale_float` compiles a new kernel (5 to 8 s), and that can happen while
   serving. With dynamic scale computation (`calculate_kv_scales`) the number of variants
   is unbounded. This checkpoint uses 1.0, so there is no issue today. A guard should fall
   back to patch 01 when q_scale != 1, or q_scale should become a runtime Float32 argument.
2. **`PRIMS_FMHA_DISABLE_BALANCED_SCHEDULING=1` is ignored** by the new path, which is
   always balanced; patch 01 honours it. The numerics should be the same because only the
   tile order changes, but the behaviour differs.
3. `scripts/serve.sh` does not mount the new file. Enabling the patch needs two extra `-v`
   mounts and the env flag. If only `flashinfer.py` is mounted and the flag is set, the
   import of `sm120_fp8_fmha_glue` fails at the first prefill. A try/except fallback would
   make this safe.
4. The in-kernel Q conversion costs about 16 to 20 us per call in the server at 8k, against
   about 10 us in the microbenchmark. The net saving is still about 80 us per call.
5. `results/bench_interleaved.txt` still prints "x10 layers" (an older run); the README
   uses x11. This is cosmetic.
6. Root-owned files under `kernels/fp8-attn-glue/.cache/vllm` (already noted by the author).

## Reproduce

```
cd /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/fp8-attn-glue
./run.sh correctness.py ; ./run.sh verify_correctness.py
./run.sh bench.py --reps 50 ; ./run.sh verify_bench.py --reps 60
# real patched function: add -v patch/b/.../flashinfer.py and -v patch/b/.../sm120_fp8_fmha_glue.py
#   over the image's vllm/v1/attention/backends/ paths, then run verify_host_overhead.py
# in-server: CONFIGS="0 1" scratchpad/verify_e2e_nsys.sh, then CONFIGS="1 0";
#   nsys export --type sqlite; python3 -I verify_nsys_attn.py ng0.sqlite ng1.sqlite
```
