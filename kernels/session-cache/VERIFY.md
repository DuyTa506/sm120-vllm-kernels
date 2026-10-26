# Verification of the session state cache (independent verifier)

**Verdict: partially confirmed.**

- **Correctness is confirmed.** I found no wrong state in concurrency, forks, abort, divergence, eviction or
  leak tests.
- **The production-config gains reproduce.**
- **The recommended deployment gains little.** `--prefix-match-unit 16` + this patch gives TTFT −7 % at c=1
  and nothing measurable at c=8.
- **Refuted: there is a severe capacity regression with the default settings** (section 4). 250 idle
  session_ids pin 46-52 % of the KV pool and are not released under prefix-cache pressure. The same
  32-agent long-session replay then re-prefills 18x more tokens: 2.6k → 47.7k per turn, with TTFT p50
  1.6 s → 69 s. A control without session_ids shows no change.
- **Three of the report's claims are overstated:**
  - that spec acceptance is "unchanged";
  - the short-session gain on top of the flag, which compared unequal workloads;
  - the c=8 production gain, which is measured against a slowed baseline.

All my scripts and results are `verify_*` files in this directory (results under `verify_results/`). Every
GPU run used its own container `session-cache-verify` inside one hold of the GPU lock (`verify_queue.sh`,
`verify_run_gpu.sh`).

## 1. Correctness

### 1a. CPU, scheduler bookkeeping

`test_scheduler_side.py` (theirs) passes when re-run. My own `verify_cpu_test.py` covers the cases theirs
does not, and all of them pass:
- **Fork.** Two concurrent adopters of one entry get the same copy-on-write sources, which are the retained
  blocks, and private, distinct destinations. The retained blocks are never in an adopter's writable set.
- **Eviction during adoption.** Dropping the entry while both adopters run causes no leak and no double free.
- **Abort.** An aborted adopter leaves the entry intact, and a retry re-adopts at the same E.
- **Pending snapshot.** Dropping an entry whose snapshot is still pending cancels the worker copy.
- **Block-aligned E.** It is retained and adopted, with no copy-on-write. The snapshot block stays
  read-only (the running state goes to a new block).
- **Preemption.** A preempted adopter re-adopts.
- The pool returns to its initial free count.

### 1b. Direct state check under concurrency (GPU, debug dump variant; their analyzer, unchanged)

**Stage A2** (`verify_state_conc.py`; `--prefix-match-unit 16`, the recommended combination, which their
stage 8 never checked). Six chains ran concurrently and produced 48 final states. Each case kind compared
with cold-prefill references:

| case kind | n | adopted | same-position layer-0 SSM distance | separated from E±1 |
|---|---:|---:|---|---:|
| chain | 24 | 18 | 0.0021-0.0026 | 24/24 |
| fork B (concurrent same-session sibling) | 6 | 6 | 0.0022-0.0025 | 6/6 |
| abort → retry | 6 | 6* | 0.0021-0.0024 | 5/6** |
| edited history, same session_id (divergence) | 6 | 0 (correct miss) | 0.0021-0.0027 | 6/6 |
| fork A | 6 | 6 | **4 of 6 at 0.18-0.30**, 2 at 0.002 | 2/6 |

\* The retry hit the stock prefix cache that the aborted request's prefill had written (`cached` ≥ the
session E), so re-adoption after an abort is covered only by the CPU test.

\*\* Layer-2 SSM noise, see stage F.

**The fork-A anomaly is explained and is not a correctness bug.** In stage A2, A and B generated the same
number of tokens and, in those 4 cases, finished in the **same scheduler step**. `on_finish(A)` creates A's
entry with a pending snapshot. `on_finish(B)` then replaces the entry (`_drop(sid)`), which **cancels A's
pending copy** and frees its blocks. A's state is therefore never written and never adoptable: B's entry
survived, and the next turn, which extends A, missed.

The debug variant had already written A's meta line. The analyzer then pairs it with a *later* dump that
reused the same block ids, so it compares an unrelated state. This is consistent with what was observed:
- B is never wrong;
- A's conv window is far from all of A's references (0.78-0.88), not just off by one.

**Stage F** (`verify_state_fork.py`, production config) is the decisive test. B finished clearly before A
(fork-early) or after A (fork-late), and the dump index and meta were copied out to verify the pairing.
All 54 states were exact, with every dump paired correctly. That includes:
- 12 fork snapshots;
- **6/6 turns that adopted the state retained after a concurrent fork**: layer-0 SSM about 0.002, conv window
  bit-identical, separated from E±1.

The 2 rows that were not separated fail only on layer-2 SSM. One is a cold first turn with no adoption,
where two identical cold prefills also differ by 0.31. That is model nondeterminism, not the patch.

### 1c. Output level

**Hot vs cold on full natural-stop generations** (`verify_hvc_full.py`, stage B, production, 8 sessions, 4
running concurrently). 56 later turns, 52 adopted. The control is a second cold run of the same prompt:

| | hot vs cold1 | cold2 vs cold1 (noise) |
|---|---:|---:|
| first-token top-5 TV, mean | 0.110 | 0.119 |
| first token equal | 50/56 | 46/56 |
| greedy LCP, median tokens | 26 | 19 |
| whole generation identical | 24/56 | 26/56 |

One outlier: session 4, turn 3, TV 0.855, a different first token, and 25 vs 759 tokens. The largest noise TV
was 0.489, and their own data has noise outliers up to 0.64-0.69. It was a fork turn, which triggered stages
E and F.

**Concurrent same-session adopters with different continuations** (`verify_fork.py`, stage E, production).
This is the copy-on-write test: a missing private copy would let B overwrite A's tail KV/state. It ran 32
fork pairs plus 32 solo adoptions, every one adopted:

| | fork A hot | fork A noise | fork B hot | fork B noise | solo A hot | solo A noise |
|---|---:|---:|---:|---:|---:|---:|
| mean TV | 0.120 | 0.135 | 0.172 | 0.174 | 0.122 | 0.123 |
| top-1 equal | 29/32 | 29/32 | 23/32 | 26/32 | 27/32 | 26/32 |
| 24/24 tokens equal | 12 | 13 | 8 | 6 | 12 | 12 |

Everything is at the noise floor. Combined with stage F, the session-4 outlier is most likely noise.

Caveat, as the author already noted: output checks cannot see a one-token offset. The direct state checks
(1b and their stage 8) are the real evidence.

### 1d. Leaks, TTL, eviction

Stage A run 1, with `VLLM_SESSION_STATE_TTL_S=150`:
- with 256 retained sessions, `vllm:kv_cache_usage_perc` was **0.456**;
- after the TTL and one poke request it was **0.0018**;
- the log shows `0 sessions, 0 retained blocks, evict_ttl 256`.

There is no block leak after about 330 session requests, about 25 LRU (`evict_max`) evictions and 256 TTL
evictions. The server logs of all 6 GPU runs contain no tracebacks or errors.

## 2. Performance (re-measured; same server per A/B, fresh salts)

| run | prefill mean | TTFT mean / p50 / p90 ms | spec accept |
|---|---:|---|---:|
| **B, production, long c=1, nosess** | 5460 | 557 / 560 / 722 | 0.954 |
| **B, production, long c=1, sess** (run first) | 2225 | 351 / 340 / 481 | 0.936 |
| B, production, short c=1 (`ignore_eos`, equal workload), nosess | 4460 | 262 / 294 / 358 | 0.952 |
| B, production, short c=1 (`ignore_eos`, equal workload), sess | 1213 | 195 / 92 / 381 | 0.915 |
| **C, `--prefix-match-unit 16`, long c=1, nosess** | 2340 | 593 / 649 / 805 | 0.933 |
| **C, `--prefix-match-unit 16`, long c=1, sess** | 2129 | 551 / 585 / 775 | 0.926 |
| C, `--prefix-match-unit 16`, long c=8, nosess | 2218 | 1674 / 1831 / 2387 | 0.943 |
| C, `--prefix-match-unit 16`, long c=8, sess | 2034 | 1612 / 1757 / 2351 | 0.941 |
| C, `--prefix-match-unit 16`, short c=1 (equal workload), nosess | 1613 | 368 / 375 / 584 | 0.920 |
| C, `--prefix-match-unit 16`, short c=1 (equal workload), sess | 1213 | 296 / 180 / 551 | 0.911 |

- **Reproduced.** Production long c=1: prefill 5460 → 2225 (they had 2415), TTFT mean −37 % (they had
  −32 %). Prefill numbers are identical to theirs where the workload is identical (5460, 2340, 2218).
- **On top of `--prefix-match-unit 16`:**
  - long c=1: −9 % prefill and −7 % TTFT mean;
  - c=8: −4 % TTFT, inside the ±30 % run-to-run noise they report, so not measurable.

  This agrees with their "about 7 % at c=1, nothing at c=8".
- **Short sessions:** the median and p50 gain is large because many turns have tiny tails. p90 does not
  improve (358 → 381 production; 584 → 551 with the flag).

## 3. Problems found in the report

1. **"Spec acceptance unchanged" is not supported by my runs.** Acceptance was lower with session_id on in
   all 5 of my same-server A/B pairs:
   - production: 0.954 → 0.936 (long c=1) and 0.952 → 0.915 (short);
   - with `--prefix-match-unit 16`: 0.933 → 0.926, 0.943 → 0.941 and 0.920 → 0.911.

   Their own pairs are mixed: 0.938/0.934, 0.884/0.894, 0.951/0.973 and others. This is consistent with the
   stale MTP drafter KV at the adopted position and the removed eagle block drop. The effect is small
   (−0.2 to −3.7 points) and affects decode speed, not TTFT, but "unchanged" overstates it.
2. **The short-session numbers on top of the flag are unequal workloads.** Natural-stop runs generated
   different completions:
   - stage 7 nosess mean 294 tokens vs sess 193;
   - stage 1b: 234 vs 196;
   - stock: 347.

   The saving on top of the flag is about one previous completion, so their "1553 → 1213, p50 230 → 128 ms"
   is inflated by the 100-token completion difference. With equal workloads (`ignore_eos`, stage C) it is
   1613 → 1213 prefill and TTFT mean 368 → 296 ms, p50 375 → 180 ms, p90 584 → 551 ms.
3. **The c=8 production gain is quoted against the patched no-session_id run** (1393 ms), which was 20 %
   slower than stock in their stage 5 (1165 ms). Against stock it is 1165 → 928 mean (−20 %) and p50
   913 → 808 (−11 %), not −33 % / −30 %. The c=32 run has no stock reference.
4. **A session miss falls back further than `--prefix-match-unit 16` alone** (`verify_fallback.py`, stage C).
   After a chain of adoptions, a history edit 50 tokens before the end lost **4241 vs 3329** and
   **2679 vs 1495** tokens in 2 of 6 sessions; the other 4 were identical. An adopted turn does not always
   leave its own prompt-tail checkpoint, so a later miss (eviction, re-rendered history) costs up to about
   1.2k extra tokens.
5. **Memory with the defaults.**
   - `VLLM_SESSION_STATE_MAX=256` × 4 blocks = 1024 of 2200 blocks: **45.6 % of the KV pool, measured**.
   - Those entries are released only by TTL (600 s), LRU or < 2 % free. "Free" counts evictable prefix-cache
     blocks, so cached prefix blocks of active sessions are evicted *before* idle session entries are.
   - An entry is only useful while its hashed prefix is still cached, otherwise `miss_attn_evicted`. It can
     therefore pin 4 blocks for nothing.
   - The README's "about 0.2 % per session" is right per session; the default cap is the problem. Stage D
     measured the effect, see section 4.
6. **The stage C hold ran 1225 s of stage plus about 150 s of startup**, a little over the 20-minute hold
   guideline. This affects my run, not their claims.

## 4. Capacity regression with many idle sessions (stages D and D2), the main refutation

**Setup.** Production config, patch on, default `VLLM_SESSION_STATE_MAX=256`, TTL 600 s. All four runs are
the same 32-agent long-session replay: 8 sessions × 4 copies, turns 20-23, consistent, session_id, fresh
salts. Each pair ran on one server:
1. replay ("before");
2. 250 tiny requests (120-token prompt, 2 output tokens), each with its own session_id, i.e. 250 idle
   sessions, as from many users or conversations within 10 minutes;
3. the same replay with fresh salts ("after").

The control D2 is identical, except that step 2 sends the requests **without** session_id, so nothing is
pinned.

| run | KV usage after step 2 | prefill / later turn (mean) | later turns re-prefilling > 50 % of the prompt | TTFT p50 | TTFT p90 | wall |
|---|---:|---:|---:|---:|---:|---:|
| D before | | 2592 | 0/96 | 1.6 s | 55.6 s | 116 s |
| **D after 250 idle sessions** | **0.520** | **47,661** | **76/96** | **69.0 s** | **88.3 s** | **309 s** |
| D2 before (control) | | 2592 | 0/96 | 1.7 s | 45.0 s | 99 s |
| D2 after 250 requests without session_id | | 2592 | 0/96 | 1.6 s | 43.4 s | 100 s |

Prefix-cache hits fell from 5.50 M to 1.17 M tokens for the same 7.49 M queried. The session cache logged
`miss_attn_evicted: 749`, and at the end it **still held 256 sessions / 1024 retained blocks**. The idle,
useless entries survived, while the active agents' full-history prefix blocks were evicted.

**Mechanism** (`session_state_cache.py`):
- `maintain()` drops entries only when `block_pool.get_num_free_blocks() < 2 %` of the pool.
- The free count includes evictable cached prefix blocks, so the pool never looks under pressure until
  the prefix cache is already gone.
- An entry whose prefix has been evicted cannot hit anyway (`miss_attn_evicted`), yet it keeps its 4 pinned
  blocks.

Stock vLLM has no such pins, which D2 shows: no change after the same requests.

This is a severe, reproducible regression for any deployment with more than about 100 distinct
session_ids active within the TTL. The README's "cannot cause OOM, but competes with the prefix cache" is
true but understates it: in this test it costs an 18x prefill increase and a 43x TTFT p50 increase.

**Mitigations, untested:**
- a much lower default `VLLM_SESSION_STATE_MAX` (e.g. about 2-4x the expected concurrent agents);
- evicting by *non-evictable* free blocks, or giving retained entries lower priority than cached prefix
  blocks;
- dropping an entry as soon as its anchor prefix block is evicted.

During stage D another container's `llama-server` (6.3 GB) was also on the GPU, outside the lock. That
can change TTFT but not prefill or hit counts, and D2 ran on a GPU with the same process present.

## 5. Not verified / out of scope

- **Chat-completions clients.** Real ones (thinking enabled, Qwen templates that drop reasoning from
  earlier turns, re-serialized tool-call JSON) may not extend the generated tokens, and then nothing hits.
  The recorded traces were made with thinking disabled and do extend. Not tested end to end.
- **Re-adoption after an abort on GPU.** The stock prefix cache won in that test; the CPU test covers it.
- **Pipeline parallelism, KV connectors, the V1 runner.** The patch disables itself for these. I checked the
  code path, not at runtime.
- **The stop clamp with `stop_token_ids` beyond 8 entries, or with `min_tokens`.** Code reading only: in both
  cases it falls back to `skip_frontier`, which is safe.

## 6. Reproduce

```bash
cd kernels/session-cache
docker run --rm $(cat .mounts) -v $PWD:/w --entrypoint python3 vllm/vllm-openai:v0.32.0 /w/verify_cpu_test.py
bash verify_queue.sh A2 B C D D2 E F   # each stage in its own GPU-lock hold (A = A2 + leak test)
python3 verify_summarize.py verify_results/{B,C,D,D2}/*.json
```

- **A** (debug, `--prefix-match-unit 16`, TTL 150): the concurrent state check and the leak test.
- **A2:** A rerun after an analyzer crash.
- **B:** production A/B and hot vs cold.
- **C:** `--prefix-match-unit 16` A/B and fallback depth.
- **D:** capacity: 250 idle session_ids.
- **D2:** the control for D: the same requests without session_id.
- **E:** fork outputs.
- **F:** debug, production, forks plus adoption after a fork.
