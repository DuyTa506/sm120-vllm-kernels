# VERIFY: cache-granularity (`--prefix-match-unit 16` on CONFIG=production)

Independent verifier. Everything marked *measured* was re-run by the verifier under the GPU lock, on the same image
(vllm/vllm-openai:v0.32.0), model and `scripts/serve.sh CONFIG=production`, with port 8121 and container
`cache-granularity-verify`. Each cell is one run. I re-ran only the production config, which is the report's
recommendation. I did not re-run optimized + unit 64, which the report itself does not recommend.

**Verdict: partially confirmed.**
- Correctness holds in every case I could construct.
- The prefill saving and the 1-agent long-context TTFT win reproduce.
- The 8-agent TTFT improvement does not reproduce.
- The memory cost is understated. Unit 16 reduces how many idle sessions stay cached. When an agent fleet's working
  set sits near the pool size, that tips the prefix cache into an LRU cascade that wipes out every session's
  cached prefix (measured: 14.5x slower).

## 1. Code facts (verified in the image)

| claim | status |
|---|---|
| `--prefix-match-unit` exists (`config/cache.py:90`, `arg_utils.py:1402`) and must divide every group's block size (`kv_cache_utils.py:878-885`) | confirmed |
| `--mamba-block-size` is overwritten in align mode (`platforms/interface.py:1102-1103`: `mamba_block_size = block_size`) | confirmed |
| `mamba_page_size_padded` only pads the page (same function, l.1105-1116) | confirmed |
| sub-block hits need align-mode Mamba with block > unit, and all managers must support fine lookup (`partial_hash_hits_enabled`) | confirmed |
| EAGLE/MTP drop is one hash unit when fine-grained, else one whole block (`eagle_proof_margin`, `FullAttentionManager.find_longest_cache_hit`) | confirmed |
| prompt-tail checkpoint at `floor(P/unit)*unit - unit` with MTP (`get_mamba_prefill_checkpoint_position`) | confirmed; measured hits equal exactly this |
| "~2,150 prefilled per turn" in the task statement does not match | confirmed: `replay_long_production.json` hit rate 0.94703 over 60,762,002 prompt tokens / 770 turns = 4,180 prefilled per turn |
| their tables match their raw result files (`summarize.py`) | confirmed |

The attention manager registers only one partial entry per prompt: the last hash boundary,
`_cache_partial_tail_block`. So a fine-grained hit exists only when the new prompt contains the previous prompt up to
`floor(P/16)*16`. Any divergence before that point falls back to the block-level hit. The `div_*` / `swap` / `trunc`
cases below show this.

## 2. Throughput / TTFT replays (measured, production, long sessions 0,1,4,7, 290 turns; their `replay_subset.py`)

| agents | config | prefilled / later turn | TTFT p50 | p90 | mean | first-turn TTFT | wall s | theirs (p50 / mean / wall) |
|---|---|---|---|---|---|---|---|---|
| 1 | production | 4,759 | 488 | 692 | 490 | 173 | 184.0 | 486 / 494 / 184 |
| 1 | + unit 16 | 1,765 | 343 | 600 | 387 | 236 | 148.7 | 330 / 377 / 147 |
| 8 | production | 4,759 | **964** | 2,505 | 1,299 | 364 | 261.7 | 1,040 / 1,374 / 261 |
| 8 | + unit 16 | 1,765 | **1,002** | 2,617 | 1,334 | 407 | 172.6 | 827 / 1,188 / 170 |

- Prefill work: reproduced exactly (4,759 -> 1,765 tokens per later turn, excess 3,017 -> 23.5). The server
  prefix-hit metric moves 0.939 -> 0.977.
- 1 agent: reproduced. Paired per turn (`verify_paired.py`), unit 16 is faster in 170/173 turns above 70k context
  (median -163 ms). It is neutral below 30k (median +32 ms, faster in 24/52), consistent with the extra split step.
  First-turn TTFT rises 173 -> 236 ms (+63 ms), which confirms their split-step cost estimate.
- 8 agents: **wall time reproduced** (-34%, i.e. +52% turns/s). **The TTFT claim did not reproduce.** My baseline
  p50 is 964 ms against their 1,040, and unit 16 is 1,002 against their 827, i.e. +4% p50, +3% mean, +4% p90. Their
  "-20% p50 / -14% mean at 8 agents" is inside the run-to-run noise. In a closed-loop replay, faster turns raise the
  request rate, so the gain shows up as throughput (session time), not TTFT. Paired by context: 30-70k turns are
  slower (median +188 ms, faster in 43/122), >70k faster (median -205 ms, faster in 206/346).

## 3. Correctness (measured)

### 3a. Needle recall, hot vs cold (`verify_correctness.py`, scored by `verify_rescore.py`)

The report's first-token TV test cannot resolve errors on this server. In my runs too, identical cold runs differed
by TV 0.03-0.39. I therefore used prompts whose answer depends on content stored exactly where a fine-grained hit is
served from. Each prompt places a random 12-digit access code inside the CoW'd partial attention block
[last block boundary, T), with filler from recorded session 7 and a chat-formatted question. A wrong or stale
attention tail block or GDN state should make the hot answer wrong, or make it repeat the *previous* prompt's code.

Cases, at 9.6k / 30.9k / 75.6k tokens:
- `ext`: plain agentic extension.
- `tiny`: extension shorter than one unit.
- `trunc`: shorter than the previous prompt, cut inside the partial block.
- `swap`: same length, with the code replaced inside the partial block before T. It must answer the *new* code, and a
  stale hit would answer the old one.
- `div_in` / `div_at` / `div_after`: divergence 5 tokens before T, exactly at T, and after the attention boundary.

Concurrency cases, all under one cache salt:
- `live`: 4 consumers hit the tail checkpoint while the producer is still decoding 400 tokens.
- `race`: producer and consumer sent at the same time.
- `siblings`: 8 prompts share a 20k prefix, sent concurrently, then 8 concurrent extensions.

| config | single-request cases: cold ok / hot ok | stale answers | concurrency cases: hot ok | errors |
|---|---|---|---|---|
| production | 21/21 / 21/21 | 0 | 13/13 | 0 |
| production + unit 16 | 21/21 / 21/21 | 0 | 13/13 | 0 |

Hit positions observed with unit 16 (`hot_cached`) behave as designed:
- `ext` / `div_after` / `live` / `siblings` hit exactly at T = floor(P/16)*16 - 16, with fine-grained hits on
  concurrent siblings.
- `tiny` hits at the warm prompt's T.
- `swap` / `div_in` / `div_at` / `trunc` fall back to the block level (6,384 / 27,664 / 72,352) and answer with the
  new content.
- `race`: the consumer got no hit at all (cached 0) while the producer was prefilling. That is correct, just no
  sharing.

The first P16 run (`results/verify_P16/correct.json`) used a raw-text question and only 14 generated tokens, which
cuts the last digit. In that run 6/21 cold answers were EOS/empty. Rescored on the code minus its last digit, it has
no stale or foreign code in any hot answer. It is superseded by `verify_P16x`, which uses a chat-formatted question
and `min_tokens`.

### 3b. Engine-level GDN state check (their `state_check.py` + dump hook)

I verified the hook build: `build/prod/orig.py` is byte-identical to the image's `qwen_gdn_linear_attn.py`, and
the hooked copy only adds the 27 hook lines. RERUN: see section 6.

## 4. Memory / capacity: the claim is understated (measured)

The report says the extra memory is "one evictable 72 MB GDN checkpoint per idle session ... never blocks admission".
That is literally true: the block size and the logged KV capacity (3,976,441 tokens) are unchanged. It hides the cost
that matters for idle agent sessions, which is how many sessions stay *cached*.

Pool arithmetic (*estimate*): the server reports 50.12 GiB of KV memory, which gives about 2.2k pool blocks of
2,128 tokens shared by the attention and the 3 GDN groups. To stay hit-able, an idle session of P tokens keeps
P/2128 attention blocks plus 3 GDN state blocks. Unit 16 adds 3 more (the tail checkpoint):
- 90k context: 46 -> 49 blocks (+6.5%).
- 10k context: 8 -> 11 blocks (+37%).

`verify_pressure.py`: N idle sessions of 88-93k tokens (distinct salts, needle in the partial block), warmed with
8 agents, then every session sends its next turn.

| test | config | sessions still hit-able | hot phase s | answers correct | KV usage when idle afterwards |
|---|---|---|---|---|---|
| 48 sessions (4.34M tokens), queried oldest-first | production | **48/48** (block-level hits, mean 87,248) | **15.5** | 48/48 | 0.0 |
| same | + unit 16 | **1/48** (47 full misses: LRU cascade) | **224.9** | 48/48 | 0.0 |
| 56 sessions (5.07M tokens), queried newest-first | production | 46/56 | 62.6 | 56/56 | 0.0 |
| same | + unit 16 | **40/56** | 84.1 | 56/56 | 0.0 |

- Unit 16 cut the number of cached idle 90k sessions from 46 to 40 (-13%).
- With 48 sessions the baseline fits just inside the pool, while unit 16 overflows it. Oldest-first access is
  exactly the round-robin order of an agent fleet. Each re-prefill then evicts the next session's head blocks, so every
  session misses and the hot phase gets 14.5x slower.
- The baseline has the same cliff at a larger session count. The finer unit moves the cliff to fewer or shorter
  sessions.
- Relevance: production at 32 agents on 100-126k sessions already uses most of the pool (32 x ~62 blocks of ~2.2k).
  This is also exactly the idle-session scenario the task highlights.
- Partial eviction is handled correctly. In the newest-first run, session 16 lost its tail checkpoint but kept its
  attention blocks, fell back to a block-level hit (87,248), and answered correctly.
- No leak: the KV usage gauge returned to 0.0 after every step, all 7 server logs show 0 Traceback/ERROR lines, and
  no request failed.

## 5. Other costs found

- **CPU hashing** (*measured*, `verify_hash_cost.py`, in the image without a GPU). Every turn is a new request, so the
  whole prompt is re-hashed (sha256 per unit) in the engine core on every turn:
  - 126k-token prompt: 4.6 ms at unit 2128, 23.0 ms at unit 16 (7,875 hashes).
  - 30k-token prompt: 1.1 ms vs 5.3 ms.

  This is pure GIL time on the engine core, not measured end to end. At 32 agents it is ~18 ms extra per long turn.
  The fine-grained Mamba lookup also probes up to P/16 hashes on a miss.
- **Extra split steps**: confirmed (+63 ms first-turn TTFT at 1 agent). This makes 10-30k contexts TTFT-neutral and
  30-70k contexts mixed at 8 agents.

## 6. 32 agents on long contexts (measured, `verify_c32.py`): not covered by the report, which stopped at 8 agents

This is the production pain point from the task (1.58 s p50 at 32 agents). Setup: 8 recorded long sessions x 4
salted copies, each starting at its first turn of >= 100k tokens, then 12 more turns. Mean later-turn context is
101.7k. Turn 0 of each agent is a cold prefill of the whole context and is excluded. The steady state below is
turns >= 3 (320 turns).

| config | later prefilled / turn | full misses | TTFT p50 | p90 | mean (turns >= 3) | wall s | server hit rate |
|---|---|---|---|---|---|---|---|
| production | 4,563 | 1 | **3,326** | 4,927 | **3,365** | 388 | 0.887 |
| + unit 16 | 2,667 | **4** | **4,396** | 5,818 | **3,921** | 318 | 0.907 |

- Throughput improves (wall -18%) and prefill work drops. **TTFT is worse**: p50 +32%, mean +17%. This is a single
  run each, so treat the size as uncertain, but the direction matches the 8-agent result (no TTFT gain) and the
  capacity result.
- One recorded turn (session 5, prompt 118,570) follows a 2,871-token decode. In the baseline, 3 of the 4 copies
  still got a block-level hit (112,784). With unit 16, all 4 copies were full misses (cached 0), each re-prefilling
  118k tokens (8-23 s TTFT).
- Mechanism, consistent with section 4 (*inference*, not traced): GDN checkpoints are released to the LRU free queue
  during prefill, while the attention blocks are held until the request ends. During a long decode under pool
  pressure, the GDN states are evicted first. The extra per-session blocks of unit 16 push the pool over the edge.

## 7. Engine-level state-check re-run (measured, their `run_state.sh` under the label `verify_S16`)

Production + unit 16, 30 recorded turns:

| metric | verifier | report |
|---|---|---|
| restored == producer's state at T (max abs diff), layers 0 and 2 | 0.0 in 30/30 for both | 30/30 |
| layer 0: restored vs cold prefill of T tokens (rel L2), mean / max | 1.35e-3 / 2.82e-3 | 1.3e-3 / 2.8e-3 |
| layer 0: off-by-16 control, min / mean | 0.133 / 0.218 | 0.133 / 0.218 |
| layer 0: hot final vs cold final, mean / max | 1.89e-3 / 2.49e-3 | 1.9e-3 / 2.5e-3 |
| layer 2 (not deterministic): restored vs cold / off-by-16 | 0.026 / 0.665 | not reported |

Reproduced. Together with the needle tests (section 3a), I found no case in which a hit returned wrong state:
- concurrent producers and consumers,
- siblings sharing a prefix,
- divergence before, at and after T,
- truncation,
- tiny extension,
- partial and full eviction.

## Summary of what holds and what does not

| report claim | verifier result |
|---|---|
| no patch needed; `--prefix-match-unit` works in v0.32 | confirmed (code + measurements) |
| prefill per later turn 4,759 -> 1,765 (long), default loses ~1.5 blocks with MTP | confirmed exactly |
| long, 1 agent: TTFT p50 486 -> 330, wall -20% | confirmed (488 -> 343, wall -19%) |
| long, 8 agents: TTFT p50 1,040 -> 827, mean -14%, wall -35% | **wall confirmed (-34%); TTFT not reproduced** (964 -> 1,002 p50, +3% mean) |
| short sessions TTFT-neutral, split-step cost +35-65 ms | consistent (+63 ms first turn; paired <30k neutral) |
| KV capacity unchanged; only one evictable 72 MB checkpoint per idle session | **misleading**: idle-session capacity -13% at 90k (46 -> 40 sessions), up to ~37% fewer blocks of headroom at 10k (*estimate*); crossing the cliff caused a full LRU cascade (15.5 s -> 225 s) |
| correctness: restored GDN state bit-identical | confirmed; plus needle tests incl. concurrency, divergence, eviction: no wrong or stale answer |
| (not tested by the report) 32 agents | long context, 32 agents: wall -18% but TTFT p50 +32% / mean +17%, more full misses after a long decode |
| (not mentioned) | CPU: per-turn prompt hashing 4.6 -> 23 ms for a 126k prompt |

**Recommendation from the verifier:** do not enable `--prefix-match-unit 16` blindly in CONFIG=production.
- It is a clear win for few agents on long contexts (1-8 agents) when the KV pool has headroom.
- At high agent counts, or with many idle long sessions near pool capacity, it raises TTFT and can trigger a cache
  cascade.
- Gate it on pool headroom: (sessions x (ctx/2128 + 6) blocks) well below ~2.2k blocks.
- Or pair it with a fix that keeps the per-session footprint unchanged, e.g. freeing the block-boundary GDN state once
  a tail checkpoint exists. That fix is untested.

Not verified: the optimized + unit 64 results, short-session TTFT, and the run-to-run noise of any cell (one run
each). The 8- and 32-agent TTFT differences may be partly noise, but in no run did a TTFT gain appear above 1 agent.

## Files (verifier)

- `verify_correctness.py`: needle hot-vs-cold cases plus concurrency.
- `verify_rescore.py`: classifies answers as ok / empty / wrong / STALE.
- `verify_pressure.py`: eviction, idle sessions, KV-usage leak check; `--order fwd|rev`.
- `verify_c32.py`: 32 agents starting at about 100k context.
- `verify_paired.py`: paired per-turn TTFT comparison.
- `verify_hash_cost.py`: CPU hashing cost.
- `verify_run.sh` / `verify_queue.sh`: one lock hold per job.
- Results: `results/verify_<label>/`. Logs: `verify_queue*.log`.

Reproduce:
```
LOCK=<gpu.lock> ./verify_queue.sh "P16x|production|correct pressure|--prefix-match-unit 16" "Px|production|correct pressure|" \
  "P16z|production|pressrev|--prefix-match-unit 16" "Pz|production|pressrev|" \
  "P16|production|long1 long8|--prefix-match-unit 16" "P|production|long1 long8|" \
  "P16y|production|c32|--prefix-match-unit 16" "Py|production|c32|"
python3 verify_rescore.py results/verify_*/correct.json
python3 verify_paired.py results/verify_P/long_c1.json results/verify_P16/long_c1.json
```
