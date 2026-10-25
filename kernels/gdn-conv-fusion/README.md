# GDN prefill: fused causal conv1d + post-conv prep

One Triton kernel replaces two in the Gated DeltaNet prefill path
(`qwen_gdn_linear_attn.py::_forward_core`):

| before | after |
|---|---|
| `causal_conv1d_fn` (`_causal_conv1d_fwd_kernel`): reads mixed_qkv, writes conv output `[T, 8192]` bf16 | `fused_conv1d_post_conv_prep` (`_fused_conv_post_conv_kernel`) |
| `fused_post_conv_prep` (`_fused_post_conv_kernel`): reads the conv output again, writes q, k (l2-normed), v, g/exp(g), beta | |

The conv output no longer goes through HBM. Per 8k-token layer call that removes 128 MB
of writes and 128 MB of reads. The fused kernel moves the same bytes as a plain
`x.copy_()`: read mixed_qkv once, write q/k/v once.

## Result (RTX PRO 6000 Blackwell, vLLM v0.32.0 image, real layer-0 weights and real-text activations)

`bench.py`: GPU time per layer call (CUDA events, eager launches, no CUDA graph; the GDN core op runs eagerly in vLLM's piecewise graphs). Each number is the median of 60 runs after 10 warmup runs, with the L2 flushed before each call.
"chain" runs 30 back-to-back calls (one request's worth of GDN layers) over 4 rotating input sets and reports time / 30.
That is the most realistic number.

| case | conv | post | **base (conv+post)** | **fused** | saved / call | copy floor | chain base → fused |
|---|---|---|---|---|---|---|---|
| 8k single seq | 196.6 | 194.5 | 376.8 | **190.4** | 186 us | 186.4 | 370.4 → **183.6** |
| 4k single seq | 108.5 | 98.3 | 153.6 | **98.3** | 55 us | 96.3 | 146.2 → **75.8** |
| 16k single seq | 383.0 | 381.0 | 755.7 | **370.7** | 385 us | 370.7 | 751.0 → **370.4** |
| varlen 8k (3000,2000,1500,1000,500,192; mixed init states) | 196.6 | 192.5 | 376.8 | **188.4** | 188 us | 188.4 | 371.8 → **184.5** |
| varlen 64 x 128 (all with initial state) | 200.7 | 192.5 | 380.9 | **192.5** | 188 us | 186.4 | 375.8 → **188.0** |

(us; full output in `bench_result.txt`; CUDA-event resolution here is about 2 us.)

* The fused kernel runs at the copy floor: ~1.43 TB/s on floor bytes, the same as `x.copy_()` of the same bytes. Little is left to gain inside this kernel.
* Why the saving is larger at 8k than at 4k: at 4k the 64 MB conv output partly stays in the 128 MB L2, so the baseline post-conv kernel was already cheaper than its DRAM cost. At 8k and above the round trip goes fully to DRAM.
* Estimated end-to-end savings (30 GDN layers per forward, from the chain numbers):
  * `optimized` config (8k prompt in one forward): 30 x 186.8 us = **~5.6 ms per 8k prefill**, about 4% of the 132.7 ms GPU span.
  * `tuned` config (8k split into 2 x 4k): 60 x 70 us = ~4.2 ms.
  * 16k prompt: ~11.4 ms.
  * Decode: **0**. The decode and spec-decode paths use `causal_conv1d_update` and are untouched. A mixed decode+prefill batch goes through the fused kernel (see below), but decode steps alone do not.
* End-to-end server check: see "E2E" below.

## Correctness

`test_correctness.py` compares against the existing vLLM path: `causal_conv1d_fn` (with the real `compute_causal_conv1d_metadata`), then `fused_post_conv_prep`. Inputs are real `conv1d.weight`, `A_log`, `dt_bias`, and activations made from real embeddings x RMSNorm x the dequantized FP8 `in_proj_qkv` / `in_proj_a/b` of layers 0 and 20, tokenized from real text. conv_state contents are real mixed_qkv rows.

It compares every output: q, k, v, g (or exp g), beta, the raw decode rows, and the **whole conv_state cache**. Each case runs with 1, 2 and 4 chunks per program.

Cases:
* 8k single sequence, with and without an initial state
* varlen batch with mixed `has_initial_state` and scattered cache lines
* very short sequences of length 1 to 4, which hit both branches of the conv_state shift update
* the DS conv-state layout (the default SD layout is used everywhere else)
* decode rows in front of prefills (`split_non_spec`)
* g in log space instead of exp space
* no metadata (grid built on the host)
* prefix-caching (APC) intermediate-state stores with 2-D `cache_indices`
* the null-block skip
* layer 20 weights
* 60 random-length sequences

**Result: bit-exact on all of them** (max abs error 0, max rel error 0, conv_state identical). Conv arithmetic matches the reference: fp32 accumulation, bf16 products in the same order, the SiLU output rounded to bf16 before the fp32 L2 norm (that rounding is what the old path stored to HBM). The L2-norm reduction runs over a `[rows, 128]` tile like the reference's.

An earlier 3-D-tile version of the kernel showed isolated 1-ulp bf16 flips in q/k (about 3 in a million elements) from a different reduction order. The 2-D tile removed them. The test still accepts at most 1 ulp in q/k and says so if that happens.

Note on the APC test: the original kernel races if the conv_state line it reads as the initial state is also an intermediate-state target of the same sequence (`initial_state_idx == block_idx_first_scheduled_token`, i.e. a non-block-aligned computed prefix). In that case neither kernel is deterministic. The test therefore uses block-aligned computed prefixes, which is what vLLM's align mode produces. The GDN prefill call site does not pass the APC block arguments at all (`IS_APC_ENABLED=False`). The APC path is implemented and tested only for parity with `causal_conv1d_fn`.

GPU-free logic check: `DEV=cpu TRITON_INTERPRET=1 QUICK=1 python3 test_correctness.py` runs small shapes in the Triton interpreter against the original Triton kernel, in fp32 because the interpreter mishandles bf16. It is also bit-exact.

## Design

* **Work split = the original conv kernel's.** The chunk axis walks the (sequence, 8-token chunk) list that `compute_causal_conv1d_metadata` already put in the GDN metadata (`batch_ptr` / `token_chunk_offset_ptr`, `nums_dict[8]`). A program never straddles two sequences, and the per-sequence work happens in the program for chunk 0, exactly as in the original:
  * read the initial conv_state, controlled by has_initial_state and cache_indices
  * write the new conv_state (both seqlen >= 3 and seqlen < 3 shift cases)
  * the APC block bookkeeping
  * the null-block early exit

  `sub=2` makes the programs for even chunks handle 16 tokens and the odd ones exit at once. That variant measured best by a small margin; `sub` is 1/2/4 and all three are tested.
* **Channel axis = 32 groups of 256 channels.** Groups 0..15 take q head h + k head h: conv, SiLU, per-row L2 norm, then store into q[h] / k[h]. Groups 16..31 take two v heads each: conv, SiLU, store into v, plus g/exp(g) and beta for those heads. q and k live in one `[2, T, H, K]` buffer, so `q = qk[0]` and `k = qk[1]` are contiguous, as FlashInfer / FLA expect.
* **Tile.** The tile is 2-D, `[2*16 rows, 128]` for `sub=2` (row = half x token). The conv window comes from 4 shifted loads of the tile, rows `t-0..t-3`. The 3 overlapping ones hit L1/L2. For chunk 0 the rows before the sequence start come from conv_state, or zero without an initial state.
* **Mixed decode+prefill batches.** When `split_non_spec` is true, the first `num_decodes` sequences write their raw SiLU(conv) output to a small `[num_decode_tokens, 8192]` buffer, because the decode recurrent kernel l2-normalizes internally. All other sequences take the fused path.
* Launches use PDL like the original conv kernel (`gdc_wait` / `gdc_launch_dependents`).
* Limits: kernel width 4, K == V a power of two, SiLU, bf16 conv_state equal to the activation dtype. The patch falls back to the old two-kernel path otherwise.

## Integration patch

`03-gdn-fused-conv-post-conv.patch` applies on top of `build/vllm` (patches 01+02), with `a/vllm/... b/vllm/...` paths. It:
* adds `vllm/model_executor/layers/mamba/ops/fused_conv_post_conv.py` (the kernel)
* adds a gated branch in `_forward_core` behind **`VLLM_GDN_FUSED_CONV=1`**, default off. With the flag off the code path is unchanged; the `split_non_spec` / `num_decode_tokens` definitions only moved up a few lines.

The patched files are also in `kernels/gdn-conv-fusion/build/vllm/...`, ready to bind-mount (see `e2e.sh`).

Caveat: the kernel JIT-compiles on the first real prefill (no autotune, one compile per specialization, about 1 s). The profile-run warmup cannot pre-compile it because the conv-state strides are only known once the KV cache exists. The original conv kernel has the same behaviour.

## E2E (server, `optimized` config + patch 03, MTP-2, prefix caching on)

`e2e.sh <flag> [tag]` starts the server and records greedy outputs (`e2e_client.py`) and single-request TTFT (`vllm bench serve`, random prompts, concurrency 1). Results are in `e2e/`.

| run | flag | 8k TTFT median | 16k TTFT median |
|---|---|---|---|
| pair 2 (distinct seeds per length, 20 prompts) | off | 211.5 ms | 400.5 ms |
| pair 2 | **on** | **199.7 ms** | **390.3 ms** |
| pair 1 (16 prompts) | off | 198.8 ms | (invalid) |
| pair 1 | **on** | **194.8 ms** | (invalid) |

* Fused is faster in every valid comparison: 8k -4.0 / -11.8 ms, 16k -10.2 ms.
* The kernel estimate is -5.6 ms per 8k and -11.4 ms per 16k. Server-to-server TTFT noise here is several ms, so the e2e numbers are consistent with that estimate. They do not measure it more precisely.
* Pair 1's 16k numbers are void: it used the same random seed for 8k and 16k, so the 16k prompts got prefix-cache hits on the 8k ones (16k TTFT came out at only 1.5x the 8k TTFT). `e2e.sh` now uses a distinct seed per length.
* Greedy outputs cannot verify bit-exactness end to end: the baseline server is itself non-deterministic. With the flag off, the same prompts give different continuations across two servers, and even when repeated within one server (MTP + async scheduling + batch-dependent kernels).
  * With the flag on, all outputs are coherent and of the same character.
  * Correctness evidence is therefore the kernel-level bit-exact test above, which covers q/k/v/g/beta and the conv_state cache.

## Reproduce

```
R=/home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels
L=<scratchpad>/gpu.lock
# correctness + microbench (one GPU hold, ~3 min)
flock $L sg docker -c "docker run --rm --name gdn-conv-fusion-run --gpus device=0 --ipc=host --entrypoint bash \
  -e HF_HUB_OFFLINE=1 -v /opt/intramind/hf-cache:/root/.cache/huggingface:ro -v $R:/work \
  -w /work/kernels/gdn-conv-fusion vllm/vllm-openai:v0.32.0 run_all.sh"
# GPU-free logic check (Triton interpreter, no lock needed)
sg docker -c "docker run --rm --entrypoint python3 -e TRITON_INTERPRET=1 -e DEV=cpu -e QUICK=1 -e HF_HUB_OFFLINE=1 \
  -v /opt/intramind/hf-cache:/root/.cache/huggingface:ro -v $R:/work -w /work/kernels/gdn-conv-fusion \
  vllm/vllm-openai:v0.32.0 test_correctness.py"
# end-to-end server A/B (optimized config), one GPU hold per run (~6-8 min each)
cd $R && flock $L sg docker -c "kernels/gdn-conv-fusion/e2e.sh 0" && flock $L sg docker -c "kernels/gdn-conv-fusion/e2e.sh 1"
```

Files: `fused_conv_post_conv.py` (kernel + wrapper), `common.py` (real-weight inputs + reference path), `test_correctness.py`, `bench.py`, `run_all.sh`, `e2e.sh` + `e2e_client.py`, `03-gdn-fused-conv-post-conv.patch`, `build/vllm/...` (patched files), `bench_result.txt`, `e2e/`.
