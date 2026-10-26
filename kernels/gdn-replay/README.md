# GDN speculative-decode state replay (Qwen3.6-35B-A3B-NVFP4, vLLM v0.32.0, SM120, MTP-2)

Savings come from not writing the 3 per-token states of every GDN layer on each verify step. Everything here is under `kernels/gdn-replay/`; run scripts: `run_kernel_tests.sh`, `e2e.sh <off|on|check|shadow|onpw>`, `ab.sh`, `make_patch.sh`.

## 1. Data flow of GDN speculative decode in vLLM v0.32.0 (what the code does today)

Deployment: Model Runner V2 (`Using V2 Model Runner` in the server log), `mamba_cache_mode=align`
(prefix caching on), mamba block 2176 tokens, MTP-2 so every verify step runs 3 positions per request,
`VLLM_GDN_DECODE_KERNEL=cuda` default, so the GDN layer calls `torch.ops._C.fused_gdn_decode_post_conv_mtp`
(`csrc/gdn/fused_gdn_decode_kernel.cu`, kernel `gdn_decode_post_conv_mtp_kernel<float,2,false>` in the
nsys traces). The FLA/Triton recurrent kernel is only the fallback for mixed prefill+decode steps.

Layer shapes (checkpoint `config.json`): 30 GDN layers, 32 value heads, 16 key heads, K = V = 128,
`mamba_ssm_dtype` fp32. Recurrent state per request and layer: 32 x 128 x 128 x 4 B = **2.0 MiB** (2,097,152 B).

**State slots.** In align mode each request owns, per layer, a window of `1 + num_speculative_blocks = 3`
consecutive block-table columns `[k, k+1, k+2]` (`k` = column of the last scheduled token,
`GDNAttentionMetadata.spec_state_indices_tensor`, built by `mamba_get_block_table_tensor` / MRV2
`compute_aligned_state_indices`). Window slot `j` holds the state after the request's j-th verified token.

**Per step and layer** (`gdn_decode_post_conv_mtp_kernel`, grid = requests x 32 value heads, one CTA per
(request, head), 4 chunks of 32 state rows):
1. `source_slot = state_indices[r, num_accepted_prev - 1]`: read the committed state from there (1 read, 2 MiB).
2. For each of the 3 tokens: h *= decay; delta = (v - h.k) * beta; h += k * delta; out = h.q.
3. After **every** token, store h to `state_indices[r, t]` (3 writes, 6 MiB), whether or not that token is accepted.

**Acceptance.** The rejection sampler returns `num_sampled` (accepted drafts + 1). `MambaHybridModelState.postprocess_state`
scatters it into `num_accepted_tokens_gpu` (GPU only, no host sync). The next step's `prepare_attn` gathers it
as `num_accepted_tokens`; the kernel uses it as `source_slot` offset and `causal_conv1d_update` as conv window
offset. Nothing is rolled back: the unaccepted slots are simply never read.

**Prefix cache / align (block 2176).** Before the forward, `preprocess_state` moves a request whose scheduled
tokens cross a block boundary to the next window (pre-copy of the committed state, slot `src_col + num_accepted - 1`).
After sampling, `run_fused_postprocess_align` copies the state **at the boundary position** (window slot
`b - c - 1`, which can be an intermediate speculative slot) into the block that just completed, so that block can
be cached. This is the one consumer of intermediate speculative states.

Bytes per step (state only, 30 layers, stock path = 1R + 3W per request-layer = 8.39 MB):

| concurrency | requests x layers | stock bytes/step | at 1.5 TB/s |
|---:|---:|---:|---:|
| 1 | 30 | 252 MB | 0.17 ms |
| 4 | 120 | 1.01 GB | 0.67 ms |
| 16 | 480 | 4.03 GB | 2.7 ms |
| 64 | 1920 | 16.1 GB | 10.7 ms |

(matches `decode-roofline`: 353 us per layer at c=64, 10.2 ms per step.)

**Upstream already ships the same idea for Kimi-K3 (KDA).** `models/kimi_k3/nvidia/ops/recoverssm.py` +
`v1/worker/gpu/model_states/recoverssm.py`: the verify kernel does not touch the checkpoint state and caches
per-token `correction` and `k, g`; `RecoverSSMState.commit_step` runs after sampling and folds the accepted tokens
in with one kernel for all layers, also producing the block-boundary state. It is wired for KDA only
(`cache_config.use_kda_recoverssm`). The design below is its GDN analogue, but it keeps the stock slot
convention instead of resetting `num_accepted_tokens` to 1 (so conv state and the align copy kernels need no change).

## 2. Design

**Idea.** The verify kernel writes the state after every one of the 3 tokens (3 x 2 MiB per request-layer) although
only the state after the accepted token (`n` = accepted drafts + 1, 1..3) is ever read again (plus, rarely, the
align-boundary state). Replay removes the three writes from the verify step:

1. **verify** (`gdn_verify`): same math as the stock kernel, but no state writes. Per token it saves `delta` (fp32,
   128 per value head), the L2-normalised `k` (fp32) and the `decay` (fp32) in per-layer scratch
   ([rows, 3, ...], about 75 KB per row and layer, 35x smaller than a state). A per-row marker `= *epoch` says
   "this row's scratch is valid for this step".
2. **commit** (`gdn_commit`, one launch per KV-cache group, after sampling, inside
   `MambaHybridModelState.postprocess_state`): for each armed row it reads the committed state, replays the first `n`
   tokens `h = h*decay; h = fma(k, delta, h)` and writes the result to window slot `n-1`. If the step crosses a
   mamba block boundary it also writes the replayed state at boundary position `m` to window slot `m-1` (the one the
   align postprocess copies into the cached block). Then the device epoch is incremented.

Bytes per request-layer: stock 1R + 3W = 8.39 MB; replay verify 1R (+0.2 MB scratch) and commit 1R + 1W (+ scratch
read), i.e. about 6.5 MB at the byte level. Microbenchmarks (below) show the verify kernel is bandwidth-bound at 1.4 TB/s
and commit as well, so the byte reduction (about 22 %) is realised as 28 % at B=64.

**Compat form, same-step (implemented).** Commit writes window slot `n-1` and leaves `num_accepted_tokens`
semantics untouched (the next step still reads `state_indices[num_accepted-1]`, conv state and the align copy kernels
are unchanged). Rows whose marker is not the current epoch (mixed prefill+decode steps, the Triton fallback, steps
below the batch threshold, batch larger than the scratch) keep exactly the stock behaviour because the stock kernel
already wrote their slots.

**Deferred form (kernel implemented, not integrated).** Skip the commit entirely and fold the previous step's `n`
tokens at the beginning of the next verify kernel (1R + 1W + scratch): 5.6 ms/step at c=64 in the microbenchmark.
It needs per-request "pending" state across steps, a flush on every non-pure step, preemption, finish, abort and
prefix-cache boundary handling, and it changes which slot holds the truth. That is too invasive to be safe; it is
the only form that reaches the ~5 ms estimate.

**Numerics / bit-exactness.** Replay performs the same fp32 operations in the same order as production
(`__fmul_rn`, `__fmaf_rn`, with `--use_fast_math` like vLLM's CMake), so results are bitwise equal, not just close.
Verified at three levels: kernel tests, a live per-layer check, and a live end-to-end shadow (section 4).

**Multiple KV-cache groups (found the hard way).** A hybrid model has several groups of GDN layers; every group has
its own block table (`spec_state_indices_tensor`) and layer i of every group shares one physical tensor. The first
integration used the first group's indices for all layers in a single commit launch and corrupted other layers'
states, collapsing MTP acceptance from 0.59 to 0.06. The commit now runs once per group with that group's metadata.

## 3. Standalone kernel results (RTX PRO 6000, CUDA graph, 8..256 distinct layers/states cycled to defeat the 128 MB L2)

Correctness (`tests/test_correctness.py`, `results/kernel_correctness_final.txt`): 205 checks, 0 failures. Bitwise
equal to the production op for out and state; n = 1, 2, 3 tokens accepted (0, 1, 2 accepted drafts), batches
1, 2, 5 ... 64, mixed query lengths, bf16/fp32 dt and norm weight, silu and sigmoid gate, verify leaves state
untouched, commit slot n-1 and boundary slot m-1 equal production, 40-step chains for the deferred kernel.

Microbenchmark (`results/kernel_bench_final.txt`), microseconds per layer call, x30 = ms per step:

| B | production | verify | commit | verify+commit | saved/layer | saved/step (30 layers) | deferred | deferred saved/step |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 9.5 | 7.9 | 2.9 | 10.8 | -1.3 | -0.04 ms | 12.4 | -0.09 ms |
| 4 | 22.8 | 9.1 | 11.7 | 20.8 | 2.0 | 0.06 ms | 15.5 | 0.22 ms |
| 8 | 46.2 | 15.4 | 23.2 | 38.6 | 7.6 | 0.23 ms | 22.8 | 0.70 ms |
| 16 | 94.8 | 28.0 | 47.0 | 75.0 | 19.8 | 0.59 ms | 49.1 | 1.37 ms |
| 32 | 196.3 | 47.8 | 94.2 | 142.1 | 54.2 | 1.63 ms | 97.9 | 2.95 ms |
| 64 | 390.0 | 93.7 | 188.0 | 281.7 | 108.2 | 3.25 ms | 202.2 | 5.63 ms |

The production op runs at 1.37-1.47 TB/s for B >= 4, so the estimate "bytes / bandwidth" is right and the saving
tracks the removed traffic: 28 % (same-step) or 48 % (deferred) at B=64. Break-even is B ~ 3.

## 4. vLLM integration and end-to-end verification

Patch: `patch/04-gdn-state-replay.patch` (relative to vLLM v0.32.0, dry-run applies; sources in `patch/b/`,
originals in `patch/a/`, regenerate with `./make_patch.sh`). New file
`model_executor/layers/mamba/gdn/gdn_state_replay.py` (+ `gdn_replay.cu`, built as a `.so`, JIT-compiled with nvcc
if the `.so` is missing), edits in `qwen_gdn_linear_attn.py` (layer registers itself, verify with stock fallback),
`mamba_hybrid.py` (record GDN metadata per group, commit in `postprocess_state`) and `envs.py`.

Flags: `VLLM_GDN_STATE_REPLAY=1` enables it (default 0: byte-identical stock path). Because the flag is registered in
`vllm/envs.py`, `envs.compile_factors()` hashes it into the compile-cache key (`results/compile_key.txt`: stock
envs.py gives the same hash for 0/1; patched gives different hashes). Debug env (unregistered, do not affect
compilation, which treats the GDN core as an opaque op): `VLLM_GDN_STATE_REPLAY_MIN_REQS` (default 8, replay only
for >= that many requests), `VLLM_GDN_STATE_REPLAY_CHECK=1|2`.

Gating: only pure speculative-decode steps with fp32 state, V2 model runner, pipeline parallel 1, batch >= MIN_REQS
and <= scratch rows; everything else runs the stock kernel unchanged.

Verification (same server config as the baselines: CONFIG=tuned, MTP-2, prefix caching, async scheduling):
- `check` mode (every layer call runs replay and stock, compares out and all 3 window states bitwise, PIECEWISE
  graphs): out mismatches 0 (the state mismatches it reported in the first run were a bug in the checker itself,
  fixed; superseded by the shadow mode).
- `shadow` mode (`CHECK=2`, `results/e2e_shadow_all.txt`): stock is authoritative and replay runs in addition in all
  30 layers; the real model-state commit result (slot n-1 of every layer, rows with 0..2 accepted drafts, group
  indices, epoch/marker bookkeeping) is compared bitwise with the stock states after every pure spec step:
  47,640+ row-layer-steps (first 8 rows of each step), 0 mismatches, 0 armed mismatches, 0 output mismatches.
  The block-boundary slot m-1 is covered by the kernel tests, and end-to-end by the prefix-cache check below.
- Prefix-cache checkpoint check (`eval/ckpt_dist.py`, `results/e2e_on.txt`): hot vs cold mean TV 0.113, top-1
  agreement 11/12, 6739 cached tokens/prompt; the repo's stock figure is 0.113 (two identical cold runs already
  differ by 0.09-0.14).
- Stress (200 requests, c=24, 37 aborted mid-generation, finish mid-step): 0 failures, 0 preemptions (preemption
  itself was not exercised on this server size), post-stress probe equals pre-stress.
- "Identical output on/off at T=0" cannot be measured end-to-end: two cold stock runs of the same prompts are not
  identical either (0/12 in `results/e2e_off.txt`; batch-invariance and MTP make greedy output nondeterministic).
  The substitute is the bitwise state/out comparison above.

Performance, alternating off/on/off/on servers, two repetitions each, 12c prompts, 128 in / 512 out
(`ab.sh`, `results/ab.txt`; median ITL ms, output tok/s):

| c | off ITL | on ITL | off tok/s | on tok/s |
|--:|--:|--:|--:|--:|
| 16 | 14.18 (14.05-14.33) | 14.34 (14.25-14.45) | 1905 | 1913 |
| 32 | 19.70 (19.57-19.76) | 19.50 (19.39-19.59) | 2629 | 2672 |
| 64 | 28.61 (28.39-28.76) | 26.23 (26.16-26.30) | 3599 | 3859 |

c=64: -2.4 ms ITL (-8.3 %), +7.2 % tokens/s, every on run faster than every off run. c=32: -0.2 ms (+1.6 % tok/s),
inside run noise. c=16: neutral. c=1 and c=4 are below the threshold (identical code path). MTP acceptance
unchanged: 0.5960 / 0.5992 off vs 0.5989 / 0.5912 on (stock figure 0.592). The microbenchmark predicted 3.25 ms at
c=64 (2.9 ms scaled to the 353 us/layer in the trace), 1.6 ms at c=32 and 0.6 ms at c=16; c=64 matches within
~15 %, c=32 and c=16 fall short because the step there is not memory-bound on the GDN state alone.

## 5. Conclusions and limitations

- The README estimate of ~5 ms/step at c=64 is only reachable with the deferred 1R+1W form (5.6 ms in the
  microbenchmark), which is judged too invasive. The safe same-step form saves 3.25 ms in the microbenchmark and
  2.4 ms measured end-to-end at c=64 (8 % ITL, 7 % throughput); nothing at c <= 16, because the saving scales with
  batch size and the extra commit pass costs more than it saves for B < 4.
- Memory: scratch of about 75 KB per row and layer (0.14 GB at 64 rows, 30 layers), no state memory added.
- Limited to: Qwen3-Next/3.5/3.6 GDN layers with the fused CUDA spec-decode kernel (`VLLM_GDN_DECODE_KERNEL=cuda`),
  fp32 state, MTP, MRV2, PP=1. TP > 1 untested (head counts are taken per rank). Preemption, KV offload and the
  decode-context paths were not exercised.
- Recommendation: keep it default-off. Enable `VLLM_GDN_STATE_REPLAY=1` (and prefer
  `VLLM_GDN_STATE_REPLAY_MIN_REQS=32`) only for deployments that routinely run >= 48 concurrent decodes; there it is
  a bit-exact, +7 % throughput change. Do not pursue the deferred form for ~2 ms more.
