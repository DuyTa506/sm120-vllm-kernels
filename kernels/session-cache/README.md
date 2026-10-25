# Session state cache: exact-position GDN/conv state reuse per agent session

Opt-in vLLM v0.32.0 patch (`VLLM_SESSION_STATE_CACHE=1`, default off). It applies to the hybrid
Qwen3.6-35B-A3B (30 GDN + 10 full-attention layers, MTP-2, prefix caching, Mamba cache mode
`align`, block 2128). When a request that carries a `session_id` finishes, the patch keeps that
session's recurrent state **at the exact final position** plus the partial last attention block.
The next turn of the same session adopts this state when its token ids extend the finished
sequence. That turn then prefills only the genuinely new suffix (tool output + chat template)
instead of everything after the last 2128-aligned checkpoint.

Files:

| file | what |
|---|---|
| `session-state-cache.patch` | unified diff `a/vllm/... b/vllm/...` against the v0.32.0 image (6 files, one new) |
| `a/`, `b/` | original and patched copies of the touched files (`b/` is what the scripts mount) |
| `.mounts` | `docker -v` flags that overlay `b/` onto the image's `dist-packages/vllm` |
| `test_scheduler_side.py` | CPU-only test of the scheduler side (hit, CoW sources, misses, block accounting) |
| `session_bench.py` | replays recorded agent sessions; per turn: prompt, cached, prefilled tokens, TTFT |
| `hot_vs_cold.py` | correctness: output after a session-state hit vs a cold full prefill, plus controls |
| `summarize.py` | markdown table from `session_bench` result files |
| `run_gpu.sh`, `stage*.sh`, `queue_all.sh` | how every number below was produced (one GPU-lock hold per stage) |
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

RESULTS_PLACEHOLDER
