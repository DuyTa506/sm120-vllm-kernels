# Session state cache: exact-position GDN/conv state reuse per agent session

Opt-in vLLM v0.32.0 patch (`VLLM_SESSION_STATE_CACHE=1`, default off). It applies to the hybrid
Qwen3.6-35B-A3B (30 GDN + 10 full-attention layers, MTP-2, prefix caching, Mamba cache mode
`align`, block 2128). When a request that carries a `session_id` finishes, the patch keeps that
session's recurrent state **at the exact final position** plus the partial last attention block.
The next turn of the same session adopts this state when its token ids extend the finished
sequence. That turn then prefills only the genuinely new suffix (tool output + chat template)
instead of everything after the last 2128-aligned checkpoint.

**Result in one paragraph (measured, details below).** On the production config, turns that hit
prefill exactly the new tokens. Mean prefill per later turn drops:

| workload | tokens prefilled | TTFT |
|---|---|---|
| long sessions, c=1 | 5460 → 2415 | mean 546 → 371 ms |
| long sessions, c=8 | 5319 → 2238 | mean 1393 → 928 ms |
| long sessions, c=32 | 5615 → 2538 | p50 3139 → 1314 ms |
| short sessions | 4172 → 1213 | p50 219 → 96 ms |

There were no preemptions and spec acceptance did not change. **However, most of that gain is also
available without this patch from vLLM's own `--prefix-match-unit 16`.** That flag's prompt-tail
checkpoint already removes the 2128-alignment and block-drop overhead (long c=1: 5460 → 2340). On
top of it, this patch only saves the previous turn's completion length per turn:
- long c=1: 2340 → 1966-2075 prefill, about 7 % lower TTFT;
- short sessions: 1379-1553 → 1213, TTFT p50 184-230 → 128 ms;
- c=8: no TTFT difference beyond run-to-run noise.

So the first step for this workload is to turn on `--prefix-match-unit 16`. This patch is a
second-order gain, best for agents whose turns generate long completions.

Correctness:
- Every snapshot checked directly against a cold-prefill reference state sits at the exact
  position (stage 8).
- Output comparisons (hot vs cold) sit at the noise floor of an ordinary prefix hit. A
  fault-injection calibration shows they cannot detect a one-token offset, which is why the
  direct state check exists.

Files:

| file | what |
|---|---|
| `session-state-cache.patch` | unified diff `a/vllm/... b/vllm/...` against the v0.32.0 image (6 files, one new) |
| `a/`, `b/` | original and patched copies of the touched files (`b/` is what the scripts mount) |
| `.mounts` | `docker -v` flags that overlay `b/` onto the image's `dist-packages/vllm` |
| `test_scheduler_side.py` | CPU-only test of the scheduler side (hit, CoW sources, misses, block accounting) |
| `session_bench.py` | replays recorded agent sessions; per turn: prompt, cached, prefilled tokens, TTFT |
| `hot_vs_cold.py` | correctness: output after a session-state hit vs a cold full prefill, plus controls |
| `state_check.py`, `state_check_analyze.py` | correctness: retained state tensors vs cold-prefill reference states at E, E-1, E+1 |
| `debug/` | debug variant of 2 patched files (snapshot meta + state dumps), used only by stage 8, **not** part of the patch |
| `fault/` | fault-injected worker (snapshot one token off), used only by stage 6 to calibrate the output check, **not** part of the patch |
| `summarize.py` | markdown table from `session_bench` result files |
| `run_gpu.sh`, `stage*.sh`, `queue_*.sh` | how every number below was produced (one GPU-lock hold per stage) |
| `results/` | raw per-turn JSON for every run, server logs in `results/logs/` |

## Why later turns prefill ~2x the new tokens

1. **Align mode.** A GDN/conv state is cached only at multiples of the 2128-token block. A turn
   that ends at position P leaves its newest reusable checkpoint at `floor(P/2128)*2128`, so on
   average about 1064 already-seen tokens are prefilled again.
2. **MTP block drop.** With eagle/MTP the scheduler also drops the last full cached block
   (`use_eagle_block_drop`, `vllm/v1/core/sched/scheduler.py:316`, used in `get_computed_blocks`).
   The drafter needs the hidden state of the position before the hit. With 2128-token blocks this
   costs another full block. In this setup it is the larger term: measured prefill on later turns
   is about 4.2k (short sessions) and 5.5k (long sessions) per turn, against about 1.2k new tokens.

Offline, from the recorded trajectories (token ids of every turn), the mean tokens prefilled per
later turn under each policy are:

| policy (estimate, offline) | short sessions | long sessions (turns 20-40) |
|---|---:|---:|
| genuinely new tokens per turn | 1177 | 1150 |
| exact-end reuse (this patch, ideal) | 909* | 1000* |
| `--prefix-match-unit 16` (eagle drop on a 16-token unit) | 1201 | 1173 |
| aligned, no eagle drop | 1909 | 2008 |
| aligned + eagle drop (stock production) | 3842 | 4075 |

\* Below "new tokens" because the next prompt re-sends the previous completion. When the model's own
generation is extended, those tokens are already in the state. Recorded generations vs replayed ones
differ, which is why the measured numbers below use the *consistent* replay mode.

## What vLLM v0.32.0 already offers (checked in the image source)

| feature | why it does not solve this |
|---|---|
| `session_id` on the request (`vllm/v1/request.py`) | carried to the engine `Request`, but unused by the scheduler and KV cache manager |
| streaming-input resumable sessions (`Request.resumable`, `scheduler.py:2466`) | keeps *all* blocks of a live request between inputs. It requires the streaming-input API, holds the full KV, and drops the last sampled token. A plain OpenAI multi-turn client never uses it. |
| `--prefix-match-unit N` (fine-grained partial hits) + `_partial_hit_reqs` / `_apply_cow` / `kv_cache_block_copies` | checkpoints the Mamba state at the **prompt** tail of each request. The previous turn's decode end is never checkpointed, so the hit ends at the previous prompt's end and the whole previous completion is prefilled again. Measured below (stage 4). |
| `enable_mamba_shared_prefix_checkpoint` (`kv_cache_manager.py:186`) | prompt junction only (shared system prompt), not per-session ends |
| KV connectors, `finalize_partial_tail_offloads` | producer prompt tail only; also disables the hybrid fast path |
| `disable_eagle_block_drop` | removes the extra block, but still aligned: about 1.9-2.0k per turn (estimate above) |
| `prefix_cache_retention_interval` | retention of already-hashed full blocks; no exact-position state |

The patch reuses the partial-hit copy-on-write path that `--prefix-match-unit` added. It is the
only piece of the existing machinery that can serve a hit ending inside a block.

## Design

Scheduler side (`b/vllm/v1/core/session_state_cache.py`, about 400 lines):

* `SessionStateCache.create(scheduler)` enables itself only when all of these hold: prefix caching,
  Mamba `align` mode, the V2 model runner, a hybrid coordinator, no KV connector, and only
  FullAttention/Mamba groups. Otherwise it logs why and stays off.
* **`on_finish(request)`** (hooked in `Scheduler._free_blocks` before the blocks are freed) runs
  for requests with a `session_id` that finish with STOPPED or LENGTH_CAPPED:
  - Retains (ref +1) the attention blocks that are not hash-registered yet: the partial last block
    and, for 2128-token blocks, usually just that one block.
  - Allocates one fresh block per Mamba group (3 here) as the snapshot destination.
  - Records the anchor block hash at `floor(n/hash_block_size)` plus the tail tokens after it, the
    cache salt and the LoRA id.
  - Queues `(req_id, use_accepted, {group: dst_block})` in `SchedulerOutput.session_state_snapshots`.
  - An idle session pins about 4 blocks. Full blocks before the tail are looked up by hash
    on reuse and are not pinned.
* **`maybe_adopt(request, ...)`** (end of `KVCacheManager.get_computed_blocks`) checks, for the same
  `session_id`: the prompt is at least as long as the finished sequence, salt and LoRA match, the
  anchor hash matches and the tail tokens match. If the stock hit is shorter than the retained
  state, it returns:
  - attention: prefix blocks found by hash with `drop_eagle_block=False`, followed by the
    retained tail blocks;
  - Mamba: `[null] * ((E-1)//B) + [snapshot]`;
  - hit length E.
  
  E is not block-aligned, so vLLM's existing partial-hit path gives the request private
  copy-on-write copies of the tail and snapshot blocks. The retained blocks stay intact for the
  next turn.
* `maintain()` (start of every `schedule()`) evicts entries by TTL (`VLLM_SESSION_STATE_TTL_S`,
  600 s), by count (`VLLM_SESSION_STATE_MAX`, 256, LRU), and under memory pressure: LRU entries are
  dropped while free blocks are below `VLLM_SESSION_STATE_MIN_FREE` (2 % of the pool).
  `reset_prefix_cache` clears everything.

Worker side (V2 model runner):

* `MambaHybridModelState.snapshot_session_states` (`b/.../model_states/mamba_hybrid.py`) is called
  from `finish_requests` before the slots are released. One Triton launch,
  `_session_snapshot_kernel`, reuses vLLM's own `_copy_mamba_state_block` (the align pre-copy body).
  It copies each finished request's running state column (`state_idx` + token bias) into the
  snapshot block, for every SSM and conv state. The result has the layout of a regular
  prefix-cache checkpoint. The copy is stream-ordered after every step already enqueued for the
  request and does no host sync.

### Which position the snapshot is at (MTP and async scheduling)

Let `n` be the request's token count. The last sampled token has no KV and no state yet.

* **No step in flight at finish** (`num_in_flight_tokens == 0`): the state covers `n-1` tokens.
  With MTP the running state after a verify step lives at column `state_idx + num_accepted - 1`
  (intermediate states per draft position). The snapshot copies exactly that column (`bias =
  num_accepted - 1`), which is what the next decode step would have read. Hit length E = n-1, so
  the next turn recomputes the last sampled token, as every prefix hit does.
* **One async step in flight** (the common case): its first input token is the last sampled token,
  so the state after that step's *first* token covers all `n` tokens. The snapshot copies offset 0
  of that step's columns, and E = n. That position's attention KV was also written by the in-flight
  step, and the block is retained. The draft tokens of that step are never used. One corner case
  is skipped: the align postprocess of that step can overwrite offset 0 when the step ends exactly
  on a 2128 boundary (`skip_align_boundary`).
* In both cases the scheduler-side condition is `num_computed_tokens - num_in_flight_tokens == n-1`.

**Stop clamp.** When EOS (or `max_tokens`) falls in the middle of a verify step, vLLM keeps the
accepted drafts after it in the state and trims them only from the output. The finished state then
covers tokens the next prompt does not contain. In the first GPU run (stage 1), about half of all
finishes were unusable for this reason. With the flag on, the V2 runner counts drafts after a stop
token, and past `max_tokens`, as rejected (`_apply_stop_clamp`, right after sampling, GPU-only).
Outputs are unchanged, since the scheduler would have trimmed those tokens anyway. The only effect
is slightly different acceptance accounting on the final step of a request. After the clamp, 199
of 200 finishes were retained (`results/logs/stage1b.out`).

**Drafter (MTP) KV.** Adopting at E reuses the drafter's KV up to E. The drafter's entry at the
last position may have been written using a token the new prompt does not have. This can only
lower draft acceptance for the first decode steps; verification keeps target outputs exact. It is
the same trade `disable_eagle_block_drop` makes. Measured acceptance did not drop (tables below).

## Measurements

### Methodology

* **Setup.** One RTX PRO 6000 Blackwell, `vllm/vllm-openai:v0.32.0`, `scripts/serve.sh`
  `CONFIG=production`: MTP-2, prefix caching, align mode, max-model-len 131072,
  `max_num_batched_tokens` 8192, gpu-mem 0.78, V2 model runner. The patched files are mounted over
  the image (`.mounts`), and `--enable-prompt-tokens-details` is set so the client sees cached
  tokens.
* **Exclusive GPU.** Every server ran inside one hold of the shared GPU lock, with nothing else on
  the GPU: `run_gpu.sh` starts the server, runs one stage script and removes the container.
* **Workload.** Recorded agent trajectories from `../../results/agentic/`
  (`trajectories_{short,long}_production.json`, token ids per turn). "Long" means sessions at
  turns 20-39, with prompts of 30-60k tokens.
* **Consistent replay.** Turn k+1 = turn k's prompt + the tokens the server *actually generated* in
  turn k + the recorded tool output and template tokens (`--consistent`), which is what a real
  agent loop sends. Replaying recorded prompts verbatim almost never extends what this server
  generates: temperature-0 generations diverge from the recorded ones (`gen_lcp` diagnostics in
  the result files). Where they do match, the recorded next prompt extends them token-exactly, so
  chat-template re-rendering is not a problem.
* **Stop modes.** "nat" turns stop at EOS (max_tokens = recorded + 64). The other turns run
  exactly the recorded completion length with `ignore_eos`.
* **Metrics.**
  - Prefill per turn = `prompt_tokens - usage.prompt_tokens_details.cached_tokens`.
  - TTFT = time to first streamed token at the client.
  - All statistics are over later turns only (turn ≥ 1 of each replayed window). Each session
    starts with a cold first turn, which is excluded.
* **A/B design.** "sess" and "nosess" run on the **same server** and differ only in whether
  `session_id` is sent. Each run uses fresh cache salts, so runs do not share cache. Stage 5 is
  stock vLLM (no patch mounted).

### Prefill tokens and TTFT per later turn

Production config (`CONFIG=production`, no `--prefix-match-unit`):

| run (stage) | conc | turns | prefill mean | prefill median | TTFT mean | TTFT p50 | TTFT p90 | spec accept | wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock vLLM, long (5) | 1 | 76 | 5460 | 5939 | 555 | 533 | 767 | 0.938 | 158 |
| patched, long, nosess (1b) | 1 | 76 | 5460 | 5939 | 546 | 532 | 733 | 0.938 | 157 |
| **patched, long, sess (1b)** | 1 | 76 | **2415** | 2761 | **371** | **351** | **559** | 0.934 | 144 |
| stock vLLM, long (5) | 8 | 152 | 5319 | 5626 | 1165 | 913 | 1820 | 0.936 | 174 |
| patched, long, nosess (2) | 8 | 152 | 5319 | 5626 | 1393 | 1159 | 2102 | 0.940 | 171 |
| **patched, long, sess (2)** | 8 | 152 | **2238** | 2622 | **928** | **808** | **1423** | 0.938 | 142 |
| patched, long ×4 copies, nosess (2) | 32 | 288 | 5615 | 5938 | 6115 | 3139 | 11602 | 0.905 | 284 |
| **patched, long ×4 copies, sess (2)** | 32 | 288 | **2538** | 2919 | **4541** | **1314** | **6164** | 0.903 | 230 |
| stock vLLM, short nat (5) | 1 | 44 | 4498 | 4401 | 340 | 360 | 472 | 0.922 | 58 |
| patched, short nat, nosess (1b) | 1 | 44 | 4172 | 4104 | 278 | 219 | 375 | 0.884 | 39 |
| **patched, short nat, sess (1b)** | 1 | 44 | **1213** | **162** | **148** | **96** | 326 | 0.894 | 29 |

With `--prefix-match-unit 16` (vLLM's own fine-grained prompt-tail checkpoints). Stage 7 repeats
stage 4 with the A/B order reversed:

| run (stage) | conc | turns | prefill mean | prefill median | TTFT mean | TTFT p50 | TTFT p90 | spec accept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| long, nosess (4) | 1 | 76 | 2340 | 3110 | 443 | 447 | 626 | 0.951 |
| long, sess (4) | 1 | 76 | 2075 | 2760 | 406 | 435 | 584 | 0.973 |
| long, sess (7) | 1 | 76 | 1966 | 2715 | 398 | 449 | 576 | 0.952 |
| long, nosess (7) | 1 | 76 | 2340 | 3110 | 424 | 458 | 599 | 0.958 |
| long, sess (4) | 8 | 152 | 1980 | 2622 | 1122 | 1011 | 1789 | 0.945 |
| long, nosess (4) | 8 | 152 | 2218 | 2924 | 989 | 833 | 1530 | 0.930 |
| long, nosess (7) | 8 | 152 | 2218 | 2924 | 1485 | 1545 | 2090 | 0.957 |
| long, sess (7) | 8 | 152 | 2007 | 2622 | 1400 | 1464 | 2075 | 0.917 |
| short nat, nosess (4) | 1 | 44 | 1379 | 692 | 244 | 184 | 427 | 0.884 |
| short nat, sess (7) | 1 | 44 | 1213 | 162 | 196 | 128 | 379 | 0.885 |
| short nat, nosess (7) | 1 | 44 | 1553 | 1082 | 227 | 230 | 373 | 0.914 |

Where the prefill goes (per later turn, from the per-turn records):
- **Session hits prefill exactly the new tail.** On short sessions the session run prefills 1213
  against a mean tail of 1212 tokens: every turn hits, 44/44. A hit at E = n-1 recomputes one
  token.
- **The remaining prefill on long sessions comes from misses.**
  - 6 of 76 long c=1 turns miss, and 8 of 152 at c=8.
  - All but one miss follows a recorded turn that was cut at the 4096-token completion cap. The
    trace's next prompt is then exactly the previous sequence with **zero** new tokens. A hit at n
    would leave nothing to compute, so it is refused (`miss_short_prompt`). This is a replay
    artifact.
  - The one other c=8 miss follows a turn whose snapshot was skipped (`skip_align_boundary`).
  - Missed turns fall back to the stock path. That costs 5.5k tokens on production and about 0.6k
    with `--prefix-match-unit 16`.
- **Breakdown per long turn (c=1):**
  - new tail after the generated tokens: 1749;
  - previous completion: 567;
  - stock production path re-prefills tail + 3711;
  - `--prefix-match-unit 16` path re-prefills tail + 591, which is the previous completion plus
    one 16-token unit dropped for MTP;
  - this patch alone re-prefills tail + 666;
  - this patch with `--prefix-match-unit 16` underneath re-prefills tail + 217-326 (stages 7 and
    4).
  - The overhead of both patch rows comes entirely from the 6 missed turns.

Retention and hit statistics (server log, `Session state cache: ...` lines):
- **Retained.** After the stop clamp, 199 of 200 finishes were retained in stage 1b, and 447 of
  450 in stage 2 (2 of the 3 losses were `skip_align_boundary`).
- **Finish path.** About 65 % of retained snapshots came from the in-flight path (one async step
  in flight at finish) and 35 % from the no-in-flight path.
- **Footprint.** About 4 blocks pinned per idle session: one attention tail block plus three Mamba
  snapshot blocks. That is about 0.2 % of the 2200-block pool each, roughly 90 MB (estimate from
  the pool size).
- **Eviction.** TTL eviction was seen working: sessions from a run 10 minutes earlier were evicted
  during stage 2 (`evict_ttl`).
- **Stats semantics.** `hit` / `miss_*` count `get_computed_blocks` calls, not requests, because a
  waiting request can be looked up more than once.

What is noise here:
- **c=8 TTFT moves ±30 % between runs of the identical config:** `--prefix-match-unit 16` nosess
  was 989 ms in stage 4 and 1485 ms in stage 7. In stage 4 the session run looked 13 % slower;
  with the order reversed in stage 7 it was 6 % faster. Neither is significant.
- **The c=8 and c=32 gains on the production config are far larger than this noise** (prefill
  -58 %, TTFT p50 -30 % / -58 %, wall -17 % / -19 %).
- **Stock vs patched with the flag off:** prefill identical (5460 / 5319); c=1 TTFT 555 vs 546 ms.
- **Stop clamp overhead.** Decode time per output token at c=1 was 1.97 ms on stock and
  1.95-2.06 ms on the patched server (both runs). The clamp's per-step cost is below this ~5 %
  noise and could not be resolved.
- **Short natural-stop runs** generate different completions run to run, even at temperature 0
  on the same server, so their "nosess" prefill varies between 4172 and 4498.

### Correctness

Three checks, from weakest to strongest.

1. **Output vs cold run** (`hot_vs_cold.py`, stages 1b and 3, 86 turns over 8 sessions including
   one long session):
   - Each turn of a session chain is answered after a session hit ("hot", 86/86 were hits, most of
     them chained on earlier adopted states). The chain alternates EOS-stopped turns with
     `ignore_eos` turns, so both snapshot paths are exercised.
   - Each hot output is compared with a cold full prefill of the same prompt, and with an ordinary
     aligned prefix hit of the same chain ("warm").
   - Metric: total-variation distance of the first-token top-20 distribution.

   | comparison | TV mean | TV quartiles (25/50/75/90 %) | 24-token greedy match (stage 3) |
   |---|---:|---|---:|
   | hot vs cold | 0.142 | 0.052 / 0.112 / 0.208 / 0.290 | 26 / 68 |
   | warm (stock prefix hit) vs cold | 0.131 | 0.036 / 0.110 / 0.181 / 0.248 | 31 / 68 |

   Hot is within the noise of a normal prefix-cache hit on this NVFP4 model: hot < warm in 38 of
   86 pairs, which a sign test does not distinguish from 50 %.

2. **Calibration: the output check cannot see a one-token offset** (stage 6, `fault/`):
   - The same cases were run on a server whose snapshot is deliberately taken one token off.
   - Hot-vs-cold TV was 0.137 (median 0.128), the same as the correct patch.
   - The client-side off-by-one control (drop one prompt token) also stayed in the noise (TV mean
     0.151).
   - So check 1 only rules out gross errors: wrong session, stale or garbage state, wrong block.

3. **Direct state check** (`state_check.py`, stage 8, debug variant `debug/`):
   - **What is compared.** Every session snapshot is logged (E, hash of its first E tokens) and its
     state block dumped for 4 GDN layers. For each one, cold requests with fresh salts are
     prefilled to E-2 … E+1 tokens and snapshotted the same way. Their state comes from the
     *prefill* kernels, which is an independent path to "the state after exactly m tokens".
   - **Coverage.** Two runs; the final run had 30 snapshots from 6 sessions × 5 turns: 23 from
     the in-flight path and 7 from the no-in-flight path; 24 were taken after an adopted hit
     (chains of up to 4 adoptions).
   - **Layers.** GDN layers 0, 1, 2 and 20. The layer-0 input is the token embedding, so
     upstream numerical noise is smallest there.
   - **Metric.** Relative L2 distance, `||snapshot - reference|| / ||reference||`. For conv, only
     the valid window is compared: the first K-1 = 3 of the block's 5 positions. The last
     `num_spec` = 2 positions are draft scratch that the align copy leaves stale; whole-block
     numbers are in `results/s8/state_check.json`.

     | GDN layer / state | session vs ref, same E (median / max) | ref vs ref, same prompt (median) | ref at E-1 (min) | ref at E+1 (min) |
     |---|---|---:|---:|---:|
     | 0 / ssm | 0.0023 / 0.0032 | 0.0000 | 0.102 | 0.103 |
     | 0 / conv window | **0.0000 / 0.0000** (bit-identical) | 0.0000 | 0.680 | 0.551 |
     | 1 / ssm | 0.013 / 0.020 | 0.002 | 0.196 | 0.175 |
     | 1 / conv window | 0.017 / 0.029 | 0.000 | 0.558 | 0.512 |
     | 2 / ssm | 0.026 / 0.035 | 0.017 | 0.431 | 0.444 |
     | 2 / conv window | 0.028 / 0.044 | 0.018 | 0.533 | 0.496 |
     | 20 / ssm | 0.072 / 0.089 | 0.070 | 0.123 | 0.122 |
     | 20 / conv window | 0.130 / 0.276 | 0.130 | 0.499 | 0.522 |

   - **Result.** In 30 of 30 snapshots the distance to the same-position reference is below the
     distances to both E-1 and E+1, for every dumped layer and both state types
     (`separated_ssm` = `separated_conv` = 30). The same-position distance is at the level of two
     cold prefills of the identical prompt, which differ through upstream nondeterminism.
   - **Conclusion.** The retained GDN and conv state is at exactly position E, for the in-flight
     path, the no-in-flight path, and after chains of adoptions.
   - **Earlier run.** The first run of this check compared whole conv blocks and showed about 0.7
     "error". The per-position breakdown traced all of it to the two scratch positions. vLLM's
     own align copy (`_copy_mamba_state_block`, `vllm/v1/worker/mamba_utils.py`) also writes only
     `conv_width - token_bias` positions and leaves the rest stale, so stock prefix-cache
     checkpoints have the same property. The scratch slots are therefore not a risk introduced
     by this patch, but they make a naive state comparison look wrong.

The scheduler side is also unit-tested on CPU (`test_scheduler_side.py`). It covers:
- an exact hit at n-1 (no step in flight) and at n (step in flight);
- that the copy-on-write sources are the retained blocks;
- misses on a changed prefix, a different salt or a changed last token;
- the frontier skip;
- that every block returns to the pool.

## How to reproduce

```bash
cd kernels/session-cache
# CPU test of the scheduler side (no GPU)
docker run --rm $(cat .mounts) -v $PWD:/w --entrypoint python3 vllm/vllm-openai:v0.32.0 /w/test_scheduler_side.py
# apply the patch to a vLLM v0.32.0 tree instead of mounting:  patch -p1 < session-state-cache.patch
# GPU stages, each in its own hold of the GPU lock (server start ~2-4 min + stage, < 18 min):
L=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
flock $L sg docker -c "bash $PWD/run_gpu.sh stage1b.sh"                         # short A/B, long c=1 A/B, hot vs cold
flock $L sg docker -c "bash $PWD/run_gpu.sh stage2.sh"                          # long c=8, c=32 A/B
flock $L sg docker -c "bash $PWD/run_gpu.sh stage3.sh"                          # hot vs cold, 68 turns
flock $L sg docker -c "bash $PWD/run_gpu.sh stage4.sh --prefix-match-unit 16"   # on top of the native flag
flock $L sg docker -c "STOCK=1 bash $PWD/run_gpu.sh stage5.sh"                  # stock vLLM reference
flock $L sg docker -c "FAULT=1 bash $PWD/run_gpu.sh stage6.sh"                  # calibration of check 1
flock $L sg docker -c "bash $PWD/run_gpu.sh stage7.sh --prefix-match-unit 16"   # stage 4 with order reversed
flock $L sg docker -c "DEBUGDUMP=1 bash $PWD/run_gpu.sh stage8.sh"              # direct state check
python3 summarize.py results/s1b/*.json results/s2/*.json results/s4/*.json results/s5_stock/*.json results/s7/*.json
```

* `run_gpu.sh` picks the first free port in 8117-8130, names the container `session-cache` and
  always removes it on exit.
* To turn the feature on in a deployment: apply the patch and set `VLLM_SESSION_STATE_CACHE=1`.
  Clients must send a stable `session_id` per agent conversation (the OpenAI-compatible request
  field already exists in v0.32). Use it together with `--prefix-match-unit 16`.
* Tunables:
  - `VLLM_SESSION_STATE_TTL_S` (default 600);
  - `VLLM_SESSION_STATE_MAX` (256 sessions);
  - `VLLM_SESSION_STATE_MIN_FREE` (fraction of the pool kept free, 0.02);
  - `VLLM_SESSION_STATE_LOG` (log the stats every N events, 100; 0 = off).

## Risks, limitations, open issues

* **Overlap with `--prefix-match-unit 16`.** That flag already captures roughly 90-95 % of the
  token savings on these traces. Long c=1: it saves 3120 of 3494 tokens per turn; short: 2793 of
  2959. This patch adds savings equal to the previous completion length,
  minus a 16-token unit, per turn.
* **The adopted hit is not 16-aligned.** `is_mamba_prefill_checkpoint_valid`
  (`vllm/v1/kv_cache_interface.py:1125`) requires `query_start % hash_block_size == 0`, so the
  prefill after an adopted hit cannot save its prompt-tail checkpoint internally.
  `_mamba_block_aligned_split` (`scheduler.py:443-556`) then stops the chunk at the next 2128
  boundary and at the prompt-tail checkpoint: one or two extra forward passes per turn. At c=8
  this eats part of the gain, and the measured TTFT difference on top of `--prefix-match-unit 16`
  is within noise.

  Fix (not done): let an adopted request skip the `next_block_boundary` stop while marking that
  boundary's Mamba block uncacheable, or start the snapshot's prefill with a ≤15-token realign
  chunk.
* **V2 model runner only.** The snapshot lives in `MambaHybridModelState`. Also unsupported: KV
  connectors, pipeline parallelism (the stop clamp is skipped under PP) and attention groups that
  are not plain full attention (sliding window, etc.). `create()` logs the reason and stays off in
  each of these cases.
* **The stop clamp changes acceptance accounting.** On the final step of a request, drafts after
  EOS or `max_tokens` count as rejected. Outputs are unchanged, since vLLM trims those tokens
  anyway. The feature needs the clamp: without it, about 50 % of finishes have extra tokens in the
  state and cannot be retained. The clamp is about 15 small torch ops per sampled step, outside
  CUDA graphs. It should be fused into one kernel; its cost was not resolvable (< ~5 %).
* **Stale drafter KV at one position.** The MTP drafter's KV at the last retained position can come
  from a token the next prompt does not have. This affects draft acceptance only; it is the same
  trade `disable_eagle_block_drop` makes. Measured acceptance was unchanged.
* **Skipped finishes.**
  - An async step in flight whose verify step ends exactly on a 2128 boundary
    (`skip_align_boundary`, 2 of 450).
  - Requests that finish by abort, by error, or with multimodal or embedding inputs.
  - A prompt that does not extend the finished sequence, for example a client that re-renders
    or trims the history. Such requests fall back to the stock path, with no error.
* **Memory.** An idle session pins about 4 blocks (about 90 MB) until TTL (600 s), LRU (256) or
  pool pressure (2 % free) evicts it. This is accounted in the block pool, so it cannot cause OOM,
  but it competes with the prefix cache. With hundreds of idle sessions, lower `VLLM_SESSION_STATE_MAX`.
* **Streaming-input (resumable) sessions** are untouched; this patch is for plain request/response
  agent loops.
* **Recorded-trace replay limits.** Truncated recorded turns (4096-token cap) produce prompts that
  do not extend the generated text; those misses are included in all numbers above.
