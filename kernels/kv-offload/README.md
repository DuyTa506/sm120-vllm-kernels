# CPU KV offloading for idle agent sessions (vLLM v0.32.0, Qwen3.6-35B-A3B hybrid, 1x RTX PRO 6000)

**Result in one line:** vLLM v0.32's *native* CPU offloading (`--kv-offloading-size 96`) already handles this hybrid
model completely: attention KV **and** GDN conv/SSM state. With 64 agent sessions of 60-126k tokens and 5-30 s tool
think time (6.1M-token working set, 1.5x the GPU's 3.95M), it turns a cache-thrashing collapse into a working
system: later-turn prefix hit 95% vs 28%, and 0.6% vs 71% of later turns recomputing their whole history. The run
completed **2.5x more later turns (in less time) and 5.8x the output tokens/s**. Restoring a 100k-token session from
host RAM moves 1.17 GB in 21-34 ms (34-55 GB/s), against 5.7 s to recompute it, and restored outputs match a cold run
within the server's own run-to-run noise. No vLLM patch is needed; the recommended change is one flag.

Everything below is either **measured** (with the file it comes from) or explicitly marked as an **estimate**.

## 1. What vLLM v0.32 offloading does with a hybrid (attention + GDN) model

Code reading (paths relative to `vllm/` inside `vllm/vllm-openai:v0.32.0`), confirmed by the runs in section 3.

| piece | where | what it does |
|---|---|---|
| `--kv-offloading-size N` (backend `native`) | `config/vllm.py:1115` | selects `OffloadingConnector` (or `SimpleCPUOffloadConnector` with env `VLLM_USE_SIMPLE_KV_OFFLOAD=1`) with `cpu_bytes_to_use = N GiB` |
| hybrid KV manager (HMA) stays on | `config/vllm.py:2170`, `offloading_connector.py:53` | both native connectors subclass `SupportsHMA`, so HMA (which this model needs) is kept |
| groups offloaded | `offloading/config.py:36` | `prefix_cacheable_group_ids`: the attention group **and the 3 GDN groups** (Mamba "align" mode makes them prefix-cacheable) |
| GDN page copy | `offloading/worker.py:126` | GDN pages are copied as raw bytes (one page = conv state + SSM state of that layer) |
| attention store | `offloading/scheduler.py:_build_store_jobs` | write-through: every completed 2128-token chunk is copied D2H while it is computed, not at eviction |
| GDN store | `offloading/scheduler.py:_build_aligned_boundary_store_jobs` | the state is stored only at the boundaries the GPU `MambaManager` hands off (`boundary_state_offloads`, `v1/core/single_type_kv_cache_manager.py:1961-2006`), i.e. the checkpoints the GPU prefix cache itself keeps |
| lookup | `offloading/scheduler.py:147, 752` | attention = longest stored prefix; GDN = a "sliding window of 1 chunk" (a hit must end where a state is stored); hits are rounded to the 2128 block (`resolve_mamba_align_size`) |
| load | `kv_offload/cpu/gpu_worker.py` | async H2D on side CUDA streams from a pinned `/dev/shm` mmap; the request waits in `WAITING_FOR_REMOTE_KVS` while the rest of the batch keeps running |
| MTP | `offloading/scheduler.py:254` | group 3 (attention + the MTP layer) is flagged as the EAGLE/MTP group: its volatile last chunk is not stored while decoding (it is stored at finish) |
| partial tail (state at a sub-block boundary) | `offloading/scheduler.py:305` | needs `--prefix-match-unit` < 2128, `blocks_per_chunk == 1` **and no EAGLE/MTP group**, so it is **off with MTP**. CPU restores land on 2128 boundaries even when the GPU cache uses a finer prefix-match unit (measured in 3.4) |

So the question "what happens to the Mamba groups if native only offloads attention" does not arise: native offloads
them. If GDN state were *not* offloaded, an attention-only restore would be useless here: the lookup requires the
GDN state at the hit end, and without it the hit falls back to 0.

Startup log (`logs/smoke_native.server.log`): `kv cache group sizes (2128, 2128, 2128, 2128)`,
`Created mmap file /dev/shm/vllm_offload_<id>.mmap (103.07 GB)`, `KV offloading: EAGLE/MTP draft attention groups [3]
detected`. Pinning the 103 GB region (cudaHostRegister) adds about 105 s to server start (216-362 s vs 149 s).

**Sizes (measured from the transfer counters).** One 2128-token block of one cache group is 11 layers x 2.167 MB =
23.84 MB: 10 attention layers + the MTP layer, while the 30 GDN layers form 3 groups padded to 11 ("Add 3 padding
layers"). In host RAM:
- attention KV: 23.84 MB / 2128 tokens = **11.2 KB/token**, i.e. 1.1 GB for a 100k-token session
- one GDN checkpoint (3 groups x 23.84 MB): **71.5 MB**
- `--kv-offloading-size 96` (103 GB) = ~4,300 blocks, i.e. ~9.2M tokens of attention KV (2.3x the GPU's 3.95M)

## 2. Test design (`idle_bench.py`)

- Sessions are slices of the 8 recorded long trajectories (`results/agentic/trajectories_long_production.json`),
  replayed token for token under distinct `cache_salt`s. Session i uses trajectory i mod 8 and a random start turn
  whose prompt is 60-120k tokens (seeded). Turn 0 of a slice is the cold arrival of the session with its history
  (mean 89k tokens). Turns 1-5 are the recorded following turns (each adds ~1k tool-result tokens plus the
  previous completion). The last prompts reach 126k.
- 64 sessions, staggered 3 s apart. Between turns each session sleeps a uniform 5-30 s (tool execution).
  First-turn context total: 5.68M tokens; last-turn total: 6.09M. GPU KV capacity: 3.95M tokens.
- Production config (`scripts/serve.sh CONFIG=production`: MTP-2, prefix caching, async scheduling, 0.78 GPU memory)
  + `--enable-prompt-tokens-details`, so each response reports `cached_tokens` (GPU + CPU hits).
- Fixed 780 s measurement window (`--duration`). At the deadline, turns still waiting for their first token enter
  the TTFT percentiles with the time waited so far (censored, a lower bound), so an overloaded configuration
  cannot look better by not finishing.
- Recorded per turn: TTFT, cached tokens, recompute = prompt - cached. Globally: deltas of `vllm:prefix_cache_*`,
  `vllm:external_prefix_cache_*`, `vllm:kv_offload_*`, and a 2 s series of KV usage.

## 3. Results

### 3.1 64 idle sessions, baseline vs native offloading (measured)

`results/w780_base.json` (no offloading) vs `results/w780_native.json` (`--kv-offloading-size 96`), seed 0, same
session set. "Later" = turns 1-5 (the turns a resuming agent makes).

| | baseline | native offload |
|---|---:|---:|
| turns completed in the window | 190 of 384 (+59 in flight at 780 s) | **384 of 384, done at 570 s** |
| later turns completed / s | 0.16 | **0.56** |
| later-turn TTFT p50 / p90 (incl. censored) | 198 s / 269 s | 2.2 s / 102 s |
| later-turn TTFT p50 / p90, turns starting after the cold wave | 190 s / 267 s (25 done + 55 censored) | **1.44 s / 3.9 s** (221 turns) |
| later turns recomputing > 8k tokens (prefix lost) | **71%** | **0.6%** (2 of 320) |
| later-turn recompute, mean tokens | 64,507 | 4,342 |
| later-turn tokens served from cache | 28% | **95%** (GPU 27%, CPU 53% of all prompt tokens) |
| output tokens/s | 31.6 | **182** |
| prompt tokens computed/s (GPU busy prefilling) | 17,709 | 12,408 |
| restores from CPU | 0 | 216 loads, 225 GB, 4.9 s total DMA (46 GB/s) |
| write-through stores | 0 | 80.8 GB, 3.9 s total DMA (21 GB/s) |
| max GPU KV usage by running requests | 24% | 96% |

Later-turn TTFT p50 by start time (`analyze.py`):

| turn start | 0-120 s | 120-240 | 240-360 | 360-480 | 480-600 | 600-720 |
|---|---:|---:|---:|---:|---:|---:|
| baseline: TTFT p50 s (mean recompute tokens) | 51 (4.5k) | 177 (65k) | 269 (92k) | 260 (91k) | 233 (84k) | 119 (censored) |
| native: TTFT p50 s (mean recompute tokens) | 55 (4.5k) | 116 (4.5k) | 23 (4.3k) | **1.6** (4.5k) | **0.5** (3.2k) | done |

What happens in the baseline: during the cold wave (5.7M prefill tokens at about 17k tok/s), the queue grows. A
waiting or thinking session's freed blocks are LRU-evicted by other prefills (the 3.95M-token cache turns over in
about 230 s), so when its next turn is scheduled the 60-120k-token history is recomputed. That recompute evicts
further sessions: a self-sustaining thrash in which the GPU spends all its time re-prefilling (17.7k tok/s
computed, 32 output tok/s). With offloading, every completed chunk already sits in host RAM, so an evicted session
costs a 20-40 ms H2D copy plus ~4k tokens of prefill. The system drains the cold wave and settles at 1.4 s p50 TTFT.

The first 120 s are identical in both runs (no eviction yet). The 55-116 s TTFTs of native during 0-240 s are
queueing behind the cold wave's 5.7M tokens of first-turn prefill, not cache misses (recompute stays at 4.5k).

### 3.2 Moderate load: 48 sessions, starts spread over 240 s (measured)

`results/w600s48_base.json` vs `results/w600s48_native.json` (seed 1, stagger 5 s, 600 s window; first-turn
context 4.25M tokens). Here the baseline does not collapse, so this shows the gain at the edge of the GPU's
capacity:

| | baseline | native offload |
|---|---:|---:|
| all 288 turns done at | 510 s | **435 s** |
| later turns losing their prefix (> 8k recompute) | 7.5% (18/240) | **0.4%** (1/240) |
| later-turn recompute, mean tokens | 10,870 | 4,344 |
| later-turn TTFT p50 / p90, all | 20.1 s / 44.6 s | 3.4 s / 38.2 s |
| later-turn TTFT p50 / p90, turns starting after the cold wave | 2.36 s / 27.1 s | **0.96 s / 2.06 s** |
| output tokens/s | 105 | 123 |
| CPU restores | - | 42 loads, 45.6 GB in 0.92 s (50 GB/s) |

The p90 during the cold wave (~40 s in both) is queueing behind 4.25M tokens of first-turn prefill; offloading
cannot help there. After it, the baseline's tail comes from the 7.5% of turns that recompute a 60-120k history
(5-8 s of GPU each, which also delays everyone queued behind them). Offloading turns those into ~25 ms copies.

### 3.3 PCIe transfer cost and restore correctness (measured, `results/pc_native.json`, `pcie_correct.py`)

Real continuations from the long trajectories: A = prompt of turn j, B = prompt of turn j+1. Paths: **cold**
(B alone), **cold2** (B again with a new salt: the server's own run-to-run noise), **gpu** (A then B, a normal GPU
prefix hit), and **cpu** (A, then `POST /reset_prefix_cache` drops the GPU cache only, then B must restore from
host RAM). Each B generates 32 greedy tokens with top-20 logprobs. Requests are sequential, so the GPU is otherwise
idle.

| len(A) | restored tokens | TTFT cold / GPU hit / CPU restore (ms) | H2D bytes | H2D time | GB/s |
|---:|---:|---:|---:|---:|---:|
| 9,997 | 6,384 | 891 / 574 / 387 | 0.14 GB | 5.9 ms | 23 |
| 25,200 | 21,280 | 956 / 393 / 424 | 0.31 GB | 13.2 ms | 23 |
| 50,116 | 46,816 | 2,012 / 465 / 465 | 0.59 GB | 10.9 ms | 55 |
| 75,023 | 72,352 | 3,436 / 296 / 322 | 0.88 GB | 25.7 ms | 34 |
| **100,195** | **97,888** | **5,710 / 387 / 420** | **1.17 GB** | **33.9 ms** | 34 |
| 99,288 | 95,760 | 5,870 / 782 / 793 | 1.14 GB | 20.8 ms | 55 |
| 120,064 | 117,040 | 8,140 / 789 / 794 | 1.38 GB | 26.5 ms | 52 |

(12 cases in the file; 2 per length.) Raw pinned-memory copy in the same container (`pcie_micro.py`): H2D 55.6 GB/s,
D2H 21.2 GB/s (PCIe 5.0 x16); 480 scattered 2.18 MB pages H2D: 53 GB/s.

- **Restoring a 100k-token session costs 21-34 ms of PCIe time (1.17 GB)**, against 5.7-5.9 s to recompute it.
  End to end, a CPU restore costs about the same as a GPU hit: CPU minus GPU TTFT ranged from -187 to +37 ms over
  the 12 cases, median +15 ms.
- Restored length = GPU-hit length in **12/12** cases (same block, same GDN checkpoint).

Correctness (mean top-20 total-variation distance of the first generated token; greedy 32-token text equality):

| pair | mean TV | max TV | top-1 equal | 32-token text equal |
|---|---:|---:|---:|---:|
| cold vs cold2 (noise floor) | 0.067 | 0.221 | 12/12 | 9/12 |
| cold vs GPU hit | 0.090 | 0.266 | 12/12 | 9/12 |
| **cold vs CPU restore** | **0.079** | 0.262 | **12/12** | **9/12** |
| GPU hit vs CPU restore | 0.086 | 0.291 | 12/12 | 8/12 |

This server is not run-to-run deterministic (two identical cold runs disagree within 32 tokens in 3/12 cases).
CPU-restored generations are indistinguishable from that noise and from normal GPU prefix hits, so there is no
sign of a wrong or stale GDN state. A corrupted SSM state would show as large TVs and wrong tokens on every
restored case, not as noise-level differences.

### 3.4 Variants (measured, 4 cases each)

- **SimpleCPUOffloadConnector** (`VLLM_USE_SIMPLE_KV_OFFLOAD=1`, `results/pc_simple.json`): also restores hybrid
  state, restored length = GPU hit 4/4, TTFT equal to native (100k: 418 ms vs GPU 378 ms). TVs are within noise
  except one 25k case (TV 0.40, top-1 and 32-token text still equal); n=4 is too small to separate it from noise.
  It uses its own metrics (`vllm:simple_kv_offload_*`), so it was not used for the main runs.
- **Native + `--prefix-match-unit 16`** (fine-grained GPU hits; `results/pc_native_pmu16.json`): both features work
  together, but the CPU restore ends at the 2128 boundary while the GPU hit ends 16-32 tokens before the prompt end:
  25,200 -> GPU 25,184 / CPU 21,280; 100,195 -> GPU 100,176 / CPU 97,888. The extra 2.3-3.9k prefill tokens cost
  +60 to +370 ms TTFT per restored turn (100k: 568-939 ms vs 507-566 ms). This is the partial-tail-with-MTP gap from
  section 1.
- **LMCache MP (lmcache 0.5.6 in the image), returns WRONG state; do not use.** `results/pc_lmcache.json`, config
  `lmcache_mp.json` (`LMCacheMPConnector`, autostarted LMCache server, `--l1-size-gb 96 --chunk-size 2128`). It
  starts and accepts the hybrid model ("Resolved LMCache MP geometry: group_tokens_per_block=[2128, 2128, 2128,
  2128]"). But with it enabled, **every reused-prefix request produces garbage**, for the plain GPU-hit path as
  well as the CPU path. Example: cold `'<tool_call>\n<function=read_file>...'` vs restored
  `'1999992ule_19951721992602171719262'`. Over 4 cases: first-token TV vs cold 0.64 (GPU hit) / 0.71 (CPU), top-1
  agreement 0/4, 32-token text equal 0/4, while cold vs cold2 stays at TV 0.03. The hits are exactly one block longer
  than vLLM's own (25,200-token prefix: 23,408 vs 21,280; 100,195: 100,016 vs 97,888). The connector only reserves
  the last token for lookup (`_reserve_last_token_for_lookup`, `lmcache_mp_connector.py:601`). It does not apply
  vLLM's MTP rule (`get_mamba_prefill_checkpoint_position`: the reusable GDN checkpoint is one block *before*
  round_down(L)). So it most likely restores a GDN page that does not hold the state at that position
  (hypothesis, consistent with the one-block offset). **The failure is MTP-specific:** the same LMCache setup
  without speculative decoding (`NOSPEC=1`; the block size is then 2096, so `lmcache_mp_nospec.json` uses
  `--chunk-size 2096`; `results/pc_lmcache_nospec.json`) restores correctly. Coherent tool calls, GPU-hit vs CPU TV
  0.015, cold vs restored TV 0.131 vs cold-cold 0.131, hits at round_down(L-1, 2096) = 25,152 / 98,512. In this
  image LMCache also needs `--kv-transfer-config` instead of `--kv-offloading-backend lmcache`, because the latter
  neither passes a size nor starts a server.

## 4. Why the restored hit ends one block early (also on the GPU)

In all 320 later turns of the native run the hit ended exactly at `round_down(previous prompt, 2128) - 2128`, for
GPU hits and CPU restores alike. This is vLLM's MTP rule, not an offloading loss:
`get_mamba_prefill_checkpoint_position` (`v1/kv_cache_interface.py:1110`) places the reusable GDN checkpoint at
`round_down(L, unit) - unit` when an EAGLE/MTP drafter is present. With the 2128-token unit, each turn therefore
recomputes ~1,064 (alignment) + 2,128 (MTP drop) tokens on top of the ~1k new ones, which matches the ~4.3k mean
recompute measured here. `--prefix-match-unit 16` shrinks this on the GPU path (another agent's area,
kernels/cache-granularity), but the CPU path keeps the 2128 unit while MTP is on (3.4).

## 5. Recommendation

Add `--kv-offloading-size 96` (native backend) to the production server. On this host, 96 GiB is 103 GB of pinned
`/dev/shm` out of 126 GB shm / 251 GB RAM. It holds ~9.2M tokens, enough for ~80 idle 110k-token sessions next to
the GPU's own ~35. Size it to what the host can spare; LRU evicts the oldest chunks when it is full.

Cost when nothing needs restoring: the write-through stores showed no measurable effect on cold prefill at
50-100k tokens. Cold TTFT on the baseline server (`results/pc_base.json`, n=1 per length) vs the offload server:
50k 2,055 vs 2,012-2,025 ms; the same 100,195-token prompt 5,732 vs 5,679-5,845 ms. For the same 25,200-token
prompt it was 826 ms vs 881-1,048 ms across the four offload-server runs. That may be a small (~10%) store cost
on short prefills or noise; the baseline side has n=1.

## 6. What is missing, and the work it would take (estimate)

Nothing blocks the idle-session use case. Remaining gaps, from code reading:
1. **Partial-tail restore with MTP** (`supports_partial_tail` is false when any EAGLE/MTP group exists,
   `offloading/scheduler.py:305`). Effect, measured: a CPU-restored turn recomputes 2.3-3.9k more tokens than a
   fine-grained GPU hit (+60-370 ms TTFT). Work: the partial-tail store/load path keys every group at the
   sub-block boundary. For the MTP attention group, the volatile last chunk (draft-layer KV of the last accepted
   position) must be excluded from the boundary key, exactly as the GPU `drop_eagle_checkpoint_block` rule does.
   That needs an MTP-aware boundary (`round_down(L, unit) - unit`, the same as `get_mamba_prefill_checkpoint_position`)
   in `_lookup` and `_build_partial_tail_store_jobs`, plus the CoW hand-off for the attention group's partial block.
   Estimate: 2-4 days, including the scheduler unit tests in `tests/v1/kv_connector/`. Payoff is small compared with
   the 5.7 s recompute that offloading already removes.
2. **LMCache MP + MTP** (lmcache 0.5.6) restores a wrong GDN state (3.4). Fix belongs in LMCache's connector: cap
   lookup/retrieve and the GDN store at vLLM's MTP checkpoint (`get_mamba_prefill_checkpoint_position` with
   `drop_eagle_block`), not at `num_tokens - 1`. Estimate: 1-2 days plus an upstream review. It is not needed,
   since native works.
3. **Startup cost**: pinning the region adds ~105 s per server start (cudaHostRegister of 103 GB).
4. **No GDN state after decode**: like the GPU cache, only prefill checkpoints are stored (turn N's prompt
   checkpoint), never a state reached while decoding. Turn N+1 therefore always restarts at turn N's prompt
   checkpoint and recomputes turn N's completion. That is the same on the GPU path; offloading neither adds nor
   removes it.

## 7. Risks

- Host RAM: 103 GB pinned and in `/dev/shm` for the server's lifetime (unlinked at start, freed at exit). Other
  containers using `/dev/shm` must fit in the remaining ~23 GB; `check_shm_free_space` refuses to start otherwise.
- Write-through D2H traffic: 80.8 GB over the 570 s run (3.9 s of DMA time). No measurable cold-TTFT cost at
  50-100k tokens; possibly ~10% at 25k (section 5).
- `VLLM_SERVER_DEV_MODE=1` was only used by the test scripts (for `/reset_prefix_cache`); do not enable it in
  production.
- Restores happen between scheduler steps. If the CPU tier misses (evicted by LRU when full), the request falls
  back to GPU hit + recompute: slower, never wrong.

## 8. Reproduce

All GPU work runs inside one hold of the shared GPU lock per configuration (container `kv-offload`, port 8121):
```
L=<gpu.lock>
# 64-session idle benchmark, 780 s window
flock $L sg docker -c "OFFLOAD=96 ./run_cfg.sh w780_native 'python3 idle_bench.py results/w780_native.json --sessions 64 --turns 6 --think 5,30 --stagger 3 --seed 0 --duration 780 --tag native'"
flock $L sg docker -c "./run_cfg.sh w780_base 'python3 idle_bench.py results/w780_base.json --sessions 64 --turns 6 --think 5,30 --stagger 3 --seed 0 --duration 780 --tag base'"
# restore cost + correctness (+ raw PCIe)
flock $L sg docker -c "OFFLOAD=96 ./run_cfg.sh correct_native 'docker exec kv-offload python3 /work/pcie_micro.py' 'python3 pcie_correct.py results/pc_native.json --targets 10000,25000,50000,75000,100000,120000 --reps 2'"
python3 analyze.py results/w780_base.json results/w780_native.json
python3 pc_summary.py results/pc_native.json
```
`queue2.sh` ... `queue6.sh` are the exact job lists used (64-session runs + correctness; baseline re-run;
variants; 48-session runs; LMCache without MTP). Files:
- `run_cfg.sh`: server start (via `scripts/serve.sh CONFIG=production`), waits for >= 80 GB of free GPU memory,
  runs the clients, saves `logs/<label>.server.log`, always removes the container. `OFFLOAD=`, `SIMPLE=1`,
  `KVT_FILE=` (e.g. `lmcache_mp.json`), `EXTRA=`.
- `idle_bench.py`, `analyze.py`: idle-session benchmark and its phase/time-bin analysis.
- `pcie_correct.py`, `pc_summary.py`, `pcie_micro.py`: restore cost, correctness, raw copy bandwidth.
- `results/`, `logs/`: all raw outputs, including server logs.
- `src/` holds the extracted vLLM and LMCache sources used for reading (not needed to reproduce).
