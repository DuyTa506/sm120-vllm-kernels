# Independent verification: gdn-conv-fusion (fused GDN prefill conv1d + post-conv)

Verifier: separate agent, 2026-10-25. I did not modify the claimant's code. My additions are
`verify_extra.py`, `verify_e2e.sh`, `verify_e2e_client.py` and `verify_qwen_gdn_linear_attn.py`
(a copy of the claimant's patched GDN file; the only change is a per-call toggle, see below).

**Verdict: CONFIRMED.** The kernel is bit-exact on every case I tried, including cases the
claimant did not test. The speedup reproduces within about 2% with the claimant's script, with
my own profiler-based timing, and in an interleaved A/B inside one live server. That server test
measured 5.5 ms saved per 8k prefill and 10.9 ms per 16k prefill, against claimed estimates of
5.6 and 11.4 ms. The minor caveats below do not change the verdict.

## 1. Code review (what I looked for)

- **Same work on both sides.** The patch replaces exactly `causal_conv1d_fn` + `fused_post_conv_prep`
  in `_forward_core`. Both paths read the same `mixed_qkv` (a strided column slice of the in_proj
  output), update the same conv_state lines and produce the same q/k/v/g/beta layouts
  (contiguous `[T,H,K]` / `[T,HV,V]`, fp32 g/beta). The benchmark baseline is exactly the server's
  two calls, with the same metadata object. The saving is real byte traffic: the conv output
  (`T x 8192` bf16) is no longer written and read back. At 8k that is 2 x 128 MB, about 180 us
  at 1.4 TB/s, which matches the measured saving.
- **Metadata.** The kernel reads `attn_metadata.batch_ptr / token_chunk_offset_ptr /
  nums_dict[8]["tot"]`. I checked that `compute_causal_conv1d_metadata` (gdn_attn.py, built from
  `non_spec_query_start_loc_cpu`) fills these eagerly, so the fused path does not depend on the
  lazy fill inside `causal_conv1d_fn`'s grid lambda.
- **Spec decode (MTP).** Spec-decode rows still go through `causal_conv1d_update` and are
  untouched. In a mixed spec+prefill batch, the non-spec rows come from `index_select`
  (contiguous x), with `a`/`b` index-selected the same way as in the old path, and no raw split.
  I tested this layout (below). The decode-only path (FULL CUDA graph) never takes the branch,
  because `num_prefills == 0`.
- **CUDA graphs / torch.compile.** `vllm::qwen_gdn_attention_core` is in the default
  `splitting_ops` (config/compilation.py:780), so the branch always runs eagerly. It allocates
  outputs with `torch.empty` and needs no graph-capture handling. Uniform decode batches captured
  as FULL graphs have `num_prefills == 0` and never reach the fused branch.
- **Patch.** `03-gdn-fused-conv-post-conv.patch` applies cleanly to a copy of `build/vllm` and
  reproduces the claimant's `build/` files byte for byte. With the flag off, the code path is
  unchanged: the moved `split_non_spec` definitions are pure.
- **Index overflow.** Token offsets into x / q / k / v / raw are int64. The conv-cache line index
  is int64. The g/beta and a/b offsets are int32, but small: at most T x 64. A 32k-token case is
  exact (below).
- **PDL.** Same `gdc_wait` / `gdc_launch_dependents` pattern as the original conv kernel.

## 2. Reproduction of the claimant's own scripts (run_all.sh, under the lock)

`test_correctness.py`: **ALL OK (bit-exact)**, 220 comparisons, 12 cases x sub 1/2/4.
GPU-free interpreter check (`TRITON_INTERPRET=1 DEV=cpu QUICK=1`): ALL OK (bit-exact).

`bench.py` (CUDA events, eager, L2 flushed per call, median of 50 after 10 warmup; chain = 30 calls
over 4 rotating input sets):

| case | base (claimed) | base (mine) | fused (claimed) | fused (mine) | chain base/fused (claimed) | chain (mine) |
|---|---|---|---|---|---|---|
| 8k single | 376.8 | 376.8 | 190.4 | 186.4 | 370.4 / 183.6 | 372.5 / 185.0 |
| 4k single | 153.6 | 149.5 | 98.3 | 102.4 | 146.2 / 75.8 | 147.0 / 74.9 |
| 16k single | 755.7 | 755.7 | 370.7 | 368.6 | 751.0 / 370.4 | 747.2 / 366.6 |
| varlen 8k | 376.8 | 374.8 | 188.4 | 186.4 | 371.8 / 184.5 | 371.9 / 183.9 |
| varlen 64x128 | 380.9 | 378.9 | 192.5 | 192.5 | 375.8 / 188.0 | 375.9 / 187.5 |

All values are us per layer call. Reproduced within about 2%.

## 3. My extra checks (`verify_extra.py`)

### Correctness

Compared against `causal_conv1d_fn` + `fused_post_conv_prep`, with real layer weights and
real-text activations. Every output (q, k, v, g, beta, raw) and the whole conv cache were
compared; **all bit-exact, max abs 0, max rel 0**:

- **Server layout:** `a`/`b` as strided halves of a `[T,64]` `ba` buffer, which is what
  `prepare_gdn_attention_core_inputs` produces. The claimant tested contiguous a/b only.
  Cases: 8k varlen with mixed initial states, and 8k single with an initial state.
- **Spec-mixed layout:** contiguous x, 1-token non-spec rows mixed with prefills, no raw split.
- **Mixed decode+prefill:** 6 decode rows and raw output, with strided a/b.
- **Production-size conv cache:** 3000 lines, high line indices.
- **`group_fastest=False` grid:** the fallback used when there are more than 65535 chunks.
  The claimant never tested it for correctness.
- **Long and other-layer inputs:** a 32k-token single sequence (stitched real activations), and
  layer 36 weights.
- **Determinism:** a rerun is bitwise identical in every case.

### JIT

A sweep of 49 realistic prefill/mixed shapes produced 4 kernel specializations, each compiled
on first use in about 0.9 s.

### Timing

Kernel times come from `torch.profiler` device durations, which have no event or launch
overhead. Server-like strided inputs, L2 scrubbed before each call:

| T | conv kernel | post kernel | base sum | fused kernel | chain30 base / fused (us/call) |
|---|---|---|---|---|---|
| 8k single | 203.2 | 177.3 | 380.5 | 191.9 | 369.3 / 182.9 |
| 8k varlen | 204.9 | 174.5 | 379.4 | 193.7 | 369.8 / 182.0 |
| 16k | 387.3 | 371.5 | 758.8 | 374.6 | 744.9 / 364.2 |
| 4k | 106.2 | 50.8 | 157.0 | 106.5 | 143.6 / 75.7 |
| 2k | 60.5 | 18.2 | 78.7 | 59.5 | 87.5 / 48.3 (CPU-launch bound) |
| 512 | 20.2 | 4.2 | 24.4 | 18.8 | 88.1 / 47.5 (CPU-launch bound) |

The post-conv kernel is cheap at T <= 4k: its input is still in the 128 MB L2. That is why the
saving shrinks for short prompts, as the claimant says. At 2k tokens and below the eager chain is
bound by host launch overhead. One fused launch replaces two Python wrappers (the conv wrapper
runs a grid lambda), which saves about 40 us of host time per layer. That only matters if the
GPU is waiting on the host.

## 4. End-to-end: interleaved A/B inside ONE server (`verify_e2e.sh`)

Server-to-server TTFT noise is large. The claimant's own two flag-off servers gave 198.8 and
211.5 ms medians at 8k, a 12.7 ms spread, which is bigger than the claimed effect. Their e2e
pairs therefore could not confirm the gain.

My test avoids that noise. I started one server with the `optimized` config (patches 01+02 +
their flags, MTP-2, prefix caching, async scheduling, same args as `e2e.sh`), using
`verify_qwen_gdn_linear_attn.py`. That file is the claimant's patched file, except that the fused
branch is enabled per call by the existence of `/vflag/fused_on`, a host-side file. This works
because the GDN op runs eagerly. Requests used unique random token-ID prompts (no prefix hits),
`max_tokens=1`, in ABBA order. Both arms were warmed first, and latency was measured on the
client.

| prompt | flag off median (IQR) | flag on median (IQR) | saved | kernel-based estimate |
|---|---|---|---|---|
| 8192 tok, n=40/arm | 151.6 ms (150.2-153.8) | 146.1 ms (144.7-148.6) | **5.5 ms** | 5.6 ms |
| 16384 tok, n=24/arm | 310.2 ms (309.3-311.7) | 299.3 ms (298.5-300.3) | **10.9 ms** | 11.4 ms |

The interquartile ranges (IQRs) do not overlap. The measured end-to-end saving matches the
microbenchmark estimate to within 0.5 ms.

These latencies are lower than the claimant's `vllm bench serve` TTFTs, because token-ID prompts
skip tokenization and the client is different. Only the difference between arms matters here.

**Logit check:** I compared prompt_logprobs over all prompt positions, prefilling the same prompt
with a different `cache_salt` each time. The server is not deterministic even with the flag
unchanged: off-vs-off max diff was 1.3-1.5 on random-token prompts. Flag on vs off gave
1.6-2.1, the same order of magnitude. Greedy continuations matched for 2 of 3 prompts; the third
differs in a byte-level token. End-to-end therefore cannot prove bit-exactness, which agrees with
the claimant's statement. The evidence for bit-exactness is the kernel-level tests in sections 2
and 3.

## 5. Caveats / problems found (none invalidates the claim)

1. **JIT on the first real prefill.** There are 4 specializations at about 0.9 s each in my
   sweep. They come from Triton integer specialization of `num_prefill_tokens`, `num_raw_*` and
   the strides. The baseline also JITs on first use: `_warmup_prefill_kernels` warms
   `fused_post_conv_prep` only with `output_g_exp=False`, not the conv kernel. So this is not a
   new class of problem, but the first few prefills after server start can stall for about 1 s
   each. A post-KV-cache warmup would remove it.
2. **The APC block-store path is dead code at this call site**, as the claimant says. Its parity
   test uses block-aligned prefixes only, and I did not test it further. If a future vLLM passes
   the `block_idx_*` arguments here, it needs a review.
3. **Small gain at the tuned config's 4k chunks.** The saving is about 70 us per call there, and
   about 40 us per call at 2k (mostly host overhead). The headline 5.6 ms applies to the
   `optimized` config, where 8k is one forward pass. The tuned-config figure of 4.2 ms is plausible
   from the chain numbers, but I did not measure it end to end.
4. **Decode: 0 ms**, confirmed by code inspection. Decode-only and spec-decode steps never take
   the branch.
5. Leftover `__pycache__` (root-owned) from container runs, both theirs and mine.

## Reproduce my checks

```
R=/home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels; L=<scratchpad>/gpu.lock
flock $L sg docker -c "docker run --rm --name gdn-conv-fusion-verify-x --gpus device=0 --ipc=host \
  --entrypoint python3 -e HF_HUB_OFFLINE=1 -v /opt/intramind/hf-cache:/root/.cache/huggingface:ro \
  -v $R:/work -w /work/kernels/gdn-conv-fusion vllm/vllm-openai:v0.32.0 verify_extra.py"
cd $R && flock $L sg docker -c "PORT=8143 kernels/gdn-conv-fusion/verify_e2e.sh <scratch>/vflag"
```
