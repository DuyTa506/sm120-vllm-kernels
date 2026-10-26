# Finer prefix-cache granularity for the hybrid model (GDN + attention)

Area: make an agentic turn's prefix-cache hit reach the end of the previous turn instead of a 2128-token boundary.
Setup: unsloth/Qwen3.6-35B-A3B-NVFP4-Fast, vLLM v0.32.0 image, 1x RTX PRO 6000 (SM120), MTP-2, prefix caching,
max-model-len 131072. All numbers below were measured on this machine unless marked *estimate*.

## TL;DR

* **No vLLM patch is needed.** vLLM v0.32 already supports prefix-cache hits inside a hybrid block:
  `--prefix-match-unit N` (any divisor of the hybrid block size) hashes the prompt every N tokens, the attention
  manager matches inside its last block, and the Mamba (GDN) manager checkpoints the state at the prompt's last
  N-aligned position and copy-on-writes it on a hit. It is off by default (the unit defaults to the block size).
* Use `--prefix-match-unit 16` with `CONFIG=production|tuned` (block 2128 = 16 x 133). For `CONFIG=optimized`
  (block 2176) the matching unit is 64 (the alignment patch 02's in-forward checkpoint needs); it is **not
  recommended yet** there (see below).
* The default is worse than "up to one block": with MTP the EAGLE rule drops one whole *block* from every hit, so a
  turn re-prefills on average 1/2 + 1 block = ~3,000 tokens it already had. Measured per later turn: **4,759 tokens
  prefilled for 1,742 new** (long sessions), 4,108 for 1,177 (short). The "~2,150" in the task statement does not
  match the replay's own hit rate (0.947 over 60.8M prompt tokens = 4,180 tokens per turn).
* With the finer unit: **1,765 prefilled for 1,742 new (+24 tokens)** with unit 16, +94 with unit 64.
  Long sessions, production config: later-turn TTFT p50 486 -> 330 ms (1 agent), 1,040 -> 827 ms (8 agents),
  mean -24% / -14%, replay wall time -20% / -35%. KV capacity unchanged (same block size); extra memory one evictable
  GDN checkpoint per idle session (72 MB).
* Correctness: the GDN state restored on a hit is **bit-identical** to the state the previous turn computed at that
  position (30/30 recorded turns, layers 0 and 2), and matches a from-scratch prefill to <= 2.8e-3 relative L2
  (chunking order), while a checkpoint one 16-token unit off would differ by 0.13-0.33.
* Cost: a hit inside a block makes the scheduler split the turn's prefill at the next block boundary and at the new
  prompt-tail checkpoint: 1-2 extra engine steps per turn, ~+35-65 ms TTFT at 1 agent (measured on first turns).
  On long contexts the ~3,000 saved prefill tokens are worth far more. On short (10-30k) sessions the production
  config's TTFT is about even (p50 -5%/-22%, mean +6%/-2% at 1/8 agents) while prefill work is still -71%.
* Optimized config: its default already avoids those splits (patch 02 exports block checkpoints inside the forward),
  so unit 64 only wins on long sessions at 1 agent (p50 300 -> 237 ms); at 8 agents long TTFT is neutral (mean
  +2%, throughput +21%) and short sessions get slower (mean 196 -> 222 ms at 1 agent, 397 -> 449 ms at 8). It also
  exposes a larger numerical deviation of the patched GDN prefill on very short chunks (below). Fixing the split
  (next steps) is the prerequisite for using a finer unit there.

## How v0.32 sets the units (read from the image, file:line in vllm/)

| knob | where | what it really does here |
|---|---|---|
| attention block size | `platforms/interface.py:_align_hybrid_block_size` (~l.1080-1120) | smallest multiple of the kernel block (16, or 64 with patch 01's `VLLM_FI_KERNEL_PAGE64`) whose attention page (1,024 B/token/layer, FP8 KV) is >= one GDN layer's state page (~2.18 MB): **2128** (production) / **2176** (optimized). A user `--block-size` can only raise it. |
| `mamba_block_size` | same function, `if mamba_cache_mode == "align": mamba_block_size = block_size` | **forced equal to the attention block** in align mode; `--mamba-block-size` is overwritten (it only matters with `mamba_cache_mode` none/other), so it cannot make GDN checkpoints finer. |
| `mamba_page_size_padded` | same function | pads the GDN page to the attention page so all groups share one block pool (0.0% at 2128, 2.26% at 2176). Memory, not granularity. |
| hash / match unit | `v1/core/kv_cache_utils.py:resolve_kv_cache_block_sizes` | `hash_block_size = --prefix-match-unit` if set, else gcd of group block sizes (= 2128). Must divide every group's block size. |
| sub-block hits | `kv_cache_utils.partial_hash_hits_enabled`, `HybridKVCacheCoordinator._cache_hit_alignment_tokens` | on when a Mamba group in align mode has block > hash unit and all managers support fine-grained lookup (FullAttentionManager and MambaManager do). Then hits land on hash-unit boundaries. |
| prompt-tail GDN checkpoint | `single_type_kv_cache_manager.py:MambaManager._cache_partial_tail_block`, `sched/scheduler.py:_mamba_block_aligned_split` | the state at `get_mamba_prefill_checkpoint_position(prompt_len, unit, eagle)` = floor(prompt/unit)*unit - unit (with MTP) is registered under that prefix's hash; the scheduler ends a chunk exactly there, or patch 02 exports it inside the forward when `is_mamba_prefill_checkpoint_valid` allows. A hit copies (CoW) the attention tail block and the GDN state block. |
| EAGLE/MTP drop | `FullAttentionManager.find_longest_cache_hit`, `eagle_proof_margin` | drops one **hash unit** when fine-grained, else one **whole block**: this is why the default loses 1.5 blocks per turn on average. |
| `enable_mamba_shared_prefix_checkpoint` | `MambaManager._cache_partial_tail_block`, scheduler junction stop | additionally checkpoints at a cross-request shared-prefix junction (needs the finer unit). Useful when many sessions share a long system prompt; not needed for turn-to-turn reuse (not measured). |
| `--kv-offloading-*`, connectors | `distributed/kv_transfer/...` | orthogonal (moving idle sessions' KV off GPU); the offloading scheduler also handles align-mode Mamba groups. Not in this area. |

`--block-size` larger than 2128/2176 is possible (e.g. 2304 = 256 x 9 for a 256 unit) but only adds Mamba padding;
there is no reason to do it since 16 / 64 already divide the default blocks.

## Measurements

Method: `replay_subset.py` replays the recorded sessions token-for-token (same prompts and completion lengths as
`bench/agentic_bench.py replay`), with `--enable-prompt-tokens-details` so every response reports
`cached_tokens`: prefilled = prompt - cached, new = prompt - previous turn's prompt (every recorded prompt extends
the previous one exactly). Long = recorded long sessions 0, 1, 4, 7 (290 turns, up to 126k tokens; sessions 0/1/7
add ~2,200 tokens per turn, session 4 ~1,000); short = all 8 short sessions (128 turns, 5-27k). 8 agents = each
session twice under different cache salts (as agentic_bench). "Later turns" = all turns except each session's
first. One run per cell; TTFT run-to-run noise on this GPU is roughly +-5% for p50 (not measured separately here).

### Long sessions (the target workload)

| agents | config | prefilled / later turn | new | excess | TTFT p50 ms | p90 ms | mean ms | replay wall s |
|---|---|---|---|---|---|---|---|---|
| 1 | production | 4759 | 1742 | 3017 | 486 | 703 | 494 | 184 |
| 1 | production + unit 16 | 1765 | 1742 | 24 | 330 | 594 | 377 | 147 |
| 1 | optimized | 4877 | 1742 | 3135 | 300 | 390 | 314 | 130 |
| 1 | optimized + unit 64 | 1836 | 1742 | 94 | 237 | 336 | 272 | 117 |
| 8 | production | 4759 | 1742 | 3017 | 1040 | 2890 | 1374 | 261 |
| 8 | production + unit 16 | 1765 | 1742 | 24 | 827 | 2448 | 1188 | 170 |
| 8 | optimized | 4877 | 1742 | 3135 | 593 | 1763 | 837 | 168 |
| 8 | optimized + unit 64 | 1836 | 1742 | 94 | 534 | 1610 | 850 | 133 |

### Short sessions (10-30k tokens)

| agents | config | prefilled / later turn | new | excess | TTFT p50 ms | p90 ms | mean ms | replay wall s |
|---|---|---|---|---|---|---|---|---|
| 1 | production | 4108 | 1177 | 2930 | 203 | 368 | 221 | 105 |
| 1 | production + unit 16 | 1201 | 1177 | 24 | 192 | 411 | 234 | 106 |
| 1 | optimized | 4184 | 1177 | 3006 | 109 | 139 | 196 | 101 |
| 1 | optimized + unit 64 | 1272 | 1177 | 94 | 123 | 172 | 222 | 103 |
| 8 | production | 4108 | 1177 | 2930 | 375 | 440 | 338 | 52 |
| 8 | production + unit 16 | 1201 | 1177 | 24 | 294 | 532 | 330 | 49 |
| 8 | optimized | 4184 | 1177 | 3006 | 136 | 303 | 397 | 44 |
| 8 | optimized + unit 64 | 1272 | 1177 | 94 | 197 | 328 | 449 | 48 |

### KV capacity and memory

| config | block | KV cache size | concurrency @131k |
|---|---|---|---|
| production (+/- unit 16) | 2128 | 3,976,441 tokens | 30.34x |
| optimized (+/- unit 64) | 2176 | 3,369,930 tokens | 25.71x |

The unit does not change block sizes, so capacity is identical with and without it (measured, both configs).
Per idle session it keeps one more cached GDN checkpoint (3 Mamba groups x 1 block = 72 MB, = 6.4k tokens of
attention KV) next to the block-boundary one, as an evictable cached block (never blocks admission; LRU-evicted
like any prefix). A hit additionally copies one attention block (24 MB) and one state block per GDN group
(3 x 24 MB) on the GPU (*estimate* <0.1 ms).

### Where the remaining TTFT goes: extra engine steps

A hit at a non-block-aligned position forces the scheduler to stop the turn's prefill at the next block boundary
(align-mode invariant "slot p holds the state after (p+1) x block tokens") and again at the new prompt-tail
checkpoint, unless patch 02 can export that checkpoint inside the forward, which `is_mamba_prefill_checkpoint_valid`
only allows when the chunk spans two block columns past its start (rare for a ~1-2k-token turn). The state check
shows hot prefills of 1-3 chunks (e.g. `[4736,6528) [6528,7744) [7744,7831)`). Measured cost at 1 agent, from cold
first turns (~1,150 tokens, same work, one extra step): 157 -> 220 ms (production, short), 176 -> 219 ms
(production, long), 86 -> 122 ms (optimized). At 8 agents each extra step waits for a full mixed batch, which is why
the optimized config (whose default already avoided splits through patch 02) gains throughput but not TTFT at 8.

## Correctness

Two checks, both on recorded turns: for (session, turn k) the cache is warmed with turn k-1 (prompt + 64 decoded
tokens, so the producer keeps running after registering its tail checkpoint, which exercises the CoW path), then
turn k is sent hot.

1. **End-to-end** (`hot_cold.py`, 38 turns, first-token top-5 TV and greedy agreement over 32 tokens): it cannot
   resolve small errors on this server. Two identical cold runs already differ by mean TV 0.09-0.14 and agree on all
   32 greedy tokens in only 8-13 of 38 turns (the server is not bit-deterministic); controls that drop or duplicate
   16 tokens before the hit point score TV 0.10-0.16, i.e. inside that noise. Hot runs were at the noise level in
   every config (default included). Gross errors (wrong block, zero state) would show as TV ~1 and are excluded.

| config | cold2 TV mean | hot TV mean | drop16 TV mean | dup16 TV mean | top-1 same hot / cold2 | 32 greedy tokens same hot / cold2 |
|---|---|---|---|---|---|---|
| production | 0.108 | 0.112 | 0.115 | 0.122 | 32 / 31 | 8 / 10 |
| production + unit 16 | 0.095 | 0.117 | 0.147 | 0.103 | 32 / 34 | 8 / 8 |
| optimized | 0.136 | 0.128 | 0.156 | 0.141 | 34 / 29 | 12 / 11 |
| optimized + unit 64 | 0.125 | 0.143 | 0.122 | 0.100 | 30 / 35 | 9 / 13 |

2. **Engine-level GDN state** (`state_check.py` + test-only dump hook `state-dump-hook.patch`, 30 turns, 6 of them
   90-122k tokens): GDN layer 0 is deterministic (identical cold runs give bit-identical states), so it isolates
   the property that matters, "the state restored at hit position T is the state of the first T tokens":

| config | mean hit / prev prompt | restored == producer state at T (bit-exact) | restored vs cold prefill of T tokens: mean / max | off-by-16 control: min / mean | hot final vs cold final: mean / max |
|---|---|---|---|---|---|
| production + unit 16 | 29679 / 29703 | 30/30 (others: no chunk ended at T) | 1.3e-03 / 2.8e-03 | 0.133 / 0.218 | 1.9e-03 / 2.5e-03 |
| optimized | 26765 / 29703 | 7/7 (others: no chunk ended at T) | 1.4e-03 / 2.2e-03 | 0.086 / 0.178 | 1.9e-03 / 6.1e-03 |
| optimized + unit 64 | 29611 / 29703 | 1/1 (others: no chunk ended at T) | 1.7e-04 / 2.1e-03 | 0.115 / 0.225 | 6.5e-03 / 4.7e-02 |

   With unit 16 every restored state is bit-identical to the state the previous turn computed at T. Against an
   independent cold prefill of the first T tokens the error is ~1e-3 (different chunk boundaries in the fp32
   recurrence); a checkpoint off by one 16-token unit would be ~50-250x larger. The final state after the hot turn
   matches a cold prefill of the whole prompt to <= 2.5e-3 (production).

**Finding for the optimized patch set (patch 02 build) + unit 64:** the restored state is right (bit-exact where
   comparable, <= 2.1e-3 otherwise), but the *final* state after the hot turn deviates more than in any other config
   in 5 of 30 turns (1.2e-2 to 4.7e-2, vs <= 2.5e-3 with production kernels and <= 6.1e-3 for the optimized default).
   All 5 are turns whose hot prefill ends in very short chunks (64-130 tokens, e.g. `[16960,17024) [17024,17137)`,
   or one 113-token chunk); the same turns with production kernels and unit 16 stay at ~1.7e-3. The default optimized
   config never runs such short prefill chunks after a hit, so this is a numerical weakness of the patched GDN prefill
   path on short chunks that unit 64 exposes, not a cache-restore error. It is still 10x below the off-by-16 control
   of those turns, and the end-to-end check sees nothing, but it should be understood (patch 02's `g_is_exp` /
   direct-output path is the first suspect) before shipping unit 64 together with patch 02.

## Recommendation

* `CONFIG=production` / `tuned`, long agentic sessions (the measured problem): add `--prefix-match-unit 16`.
  Clear TTFT win at 1 and 8 agents, prefill work -63% (long) / -71% (short), no capacity cost. On short sessions it
  is TTFT-neutral.
* `CONFIG=optimized`: keep the default for now. Unit 64 helps long sessions at 1 agent only and costs TTFT on short
  sessions, because every hit then needs extra scheduler steps that the default (patch 02) avoids, and the short
  prefill chunks it creates show the larger final-state deviation described under Correctness.

## Not done / next steps

* Remove the extra steps: let the prompt-tail checkpoint be exported inside the forward (patch 02's FlashInfer
  `state_checkpoints`) also when the chunk does not span two block columns. That needs a spare block-table column
  for the checkpoint block in `MambaManager.allocate_new_blocks` and a matching rule in
  `is_mamba_prefill_checkpoint_valid` / `compute_mamba_prefill_checkpoints` (scheduler, manager and worker must
  decline in lockstep). *Estimate*: -35-65 ms per turn at 1 agent and most of the 8-agent TTFT gap. Not attempted:
  the bookkeeping is subtle and a mismatch silently caches wrong state.
* Reusing the previous *completion* (only possible live, not in replay: replayed completions differ from the recorded
  ones) would need a checkpoint at the end of decode; with MTP the state at an exact unit boundary exists only
  transiently in the speculative state slots.
* Idle-session behaviour under memory pressure (think time) was not measured; the unit only adds one evictable
  72 MB checkpoint per session.

## Reproduce

```
LOCK=/tmp/.../gpu.lock   # the shared GPU lock
cd kernels/cache-granularity
# replays (+ e2e hot/cold) for one config, one lock hold each (~10 min):
LOCK=$LOCK ./queue.sh "P16_A|production|long1 hotcold long8|--prefix-match-unit 16" \
                      "P_A|production|long1 hotcold long8|" \
                      "P16_B|production|short1 short8 ckpt|--prefix-match-unit 16" ...
# engine-level state check (mounts the dump-hooked GDN file; prod = production, opt = optimized):
python3 make_dump_build.py
LOCK=$LOCK ./queue_state.sh "S16v2|prod|--prefix-match-unit 16" "SO64|opt|--prefix-match-unit 64"
python3 summarize.py      # tables from results/
```

Files: `replay_subset.py` (subset replay with per-request cached tokens, think time option), `hot_cold.py`
(end-to-end check), `state_check.py` + `make_dump_build.py` + `state-dump-hook.patch` (test-only GDN state dump,
inert unless `VLLM_CG_DUMP_DIR` is set and `$VLLM_CG_DUMP_DIR/ENABLE` exists), `run_config.sh` / `run_state.sh`
(one lock hold: start server on a free port 8121-8130, measure, remove container), `queue*.sh`, `summarize.py`,
`results/<label>/` (raw per-turn JSON, server logs). `results/old_*` are superseded first versions of the checks (the first state check compared only final states,
and its conv-state comparison was invalid; the first hot/cold check had 10 turns and no controls).

## Risks

* Upstream code path: fine-grained hybrid hits are new in v0.32 and off by default; the state check covers the
  production kernels and the optimized patch set on 30 turns, single-request steps only. Concurrent producers and
  consumers of the same prefix (several agents sharing a long prefix under one salt) were not state-checked.
* MTP: the hit is always one unit short of the previous prompt (16 or 64 tokens), by design.
* The extra split steps add ~35-65 ms at 1 agent; on short contexts this cancels the prefill saving (production)
  or makes TTFT worse (optimized).
* Optimized + unit 64: final-state deviation up to 4.7e-2 on turns ending in 64-130-token prefill chunks (see
  Correctness); not root-caused.
