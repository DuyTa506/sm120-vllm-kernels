# Independent verification: native CPU KV offloading (kernels/kv-offload)

Verifier: separate agent. I added only `verify_*` files (scripts, `verify_results/`, `verify_logs/`) and this note.
Container `kv-offload-verify`, port 8123. Every GPU job ran inside one `gpu.lock` hold, and each container was
removed before its hold ended. After every run, `/dev/shm` held 0 `vllm_offload_*` files.

**Verdict: PARTIALLY confirmed.** The headline result reproduces with a different seed. Restores are correct on
every path I tried, including paths the original tests never reached. "One flag fixes it" holds only if the CPU
tier is sized to the whole working set. The README's capacity statement overstates what the tier holds, and its
"no patch needed" also means divergent sessions get no reuse at all.

## 1. Re-run of the main measurement (seed 2 instead of seed 0, 700 s window)

Same `idle_bench.py`: 64 sessions, 60-126k tokens, 5-30 s think time, 6 turns, production config, same
workload in all three runs (5.63M first-turn tokens).

| | baseline | `--kv-offloading-size 96` | **64 GiB (new)** |
|---|---:|---:|---:|
| turns done | 168/384 at 700 s (+59 in flight) | 384/384 at 585 s | 384/384 at 645 s |
| later turns losing prefix (>8k recompute) | 70% (74/106) | 0.6% (2/320) | 4.1% (13/320) |
| later-turn cached fraction | 29% | 95% | 92% |
| later TTFT p50 / p90 (all, incl. censored) | 175 s / 265 s | 2.9 s / 103 s | 6.3 s / 102 s |
| later TTFT p50 / p90 after the cold wave | n/a (wave never drained) | 1.36 s / 4.4 s (222 turns) | 2.1 s / 10.5 s (219) |
| output tok/s | 23.9 | 185 | 168 |
| CPU loads / stores | - | 245 GB / 80 GB | 231 GB / 93 GB |

Files: `verify_results/w700s2_{base,native,off64}.json`. This matches the original numbers: 198 s / 2.2 s p50,
71% / 0.6% lost, 31.6 / 182 tok/s. **The main claim is confirmed for this workload.**

## 2. Correctness (hot vs cold), including paths the original tests did not exercise

Method as in `eval/ckpt_dist.py`: first-token top-20 TV, top-1 agreement and 32-token greedy text equality,
with cold-vs-cold2 on the same server as the noise floor.

**The server is not deterministic.** Identical sequential cold runs reach TV 0.37 and flip top-1 on near-ties.
These checks therefore catch gross state corruption, as they did for LMCache, but cannot rule out subtle
corruption below that noise.

`verify_seq.py` (idle GPU, 6 cases of 30k/60k/100k, `verify_results/seq.json`):

| path | mean TV vs cold | top-1 | text eq |
|---|---:|---:|---:|
| B cold2 (noise) | 0.044 | 6/6 | 4/6 |
| B GPU hit | 0.035 | 5/6 | 5/6 |
| B restored from CPU | 0.043 | 5/6 | 5/6 |
| B after CPU-restored divergent prompt (meant as mixed GPU+CPU; it was all CPU, see below) | 0.041 | 6/6 | 5/6 |
| A exact retry restored from CPU (A cold2 noise: 0.053, 6/6, 4/6) | 0.066 | 5/6 | 3/6 |
| C divergent prompt, 2nd send (C cold2 noise: 0.206, 4/6, 2/6) | 0.228 | 5/6 | 2/6 |

- **The CPU-restored B text equals the GPU-hit B text in 6/6 cases**, which is strong evidence the restored state
  matches the GPU state. The one top-1 miss for both B paths is a case where cold vs cold2 also has TV 0.26.
- The single outlier, an A retry at 60k with TV 0.30 and different top-1, was re-tested 4x per group
  (`verify_repeat.py`, `verify_results/repeat.json`). That prompt is a near-tie ("Now" vs `<tool_call>`): cold
  runs flip 2/4, GPU retries flip 1/4, CPU retries flip 2/4. On the other 3 prompts, mean pairwise TV is the same
  for cold-cold, cold-gpu and cold-cpu (e.g. 100k: 0.009 / 0.006 / 0.007). This is noise, not an offload effect.
- **MTP interaction:** spec-decode acceptance on restored requests was 1.00 (cpuB) and 0.98 (cpuA), against
  0.98-0.99 cold. A wrong MTP-layer KV would show up as lower acceptance; it does not.

**Under concurrent load with real CPU restores** (`verify_load.py` run alongside a 48-session `idle_bench`
background, 96 GiB, `verify_results/load96_*.json`): there were 9.3M external hit tokens and 113 GB loaded in
the window. All 16 probes hit, ending at rd(lenA)-2128. Hot vs cold: mean TV 0.030 against 0.035 noise, top-1
16/16 against 16/16, text 14/16 against 15/16. No errors and no crashes. A partial probe run (8/16, cut by my time
limit) under a collapsing 64 GiB load (`load64_*`) also showed no wrong output. **Not separable:** the per-request
source (GPU vs CPU) is not reported, so I cannot say which probe hits came from CPU.

**Correctness verdict:** no evidence of wrong state on CPU, GPU, retry, divergent or concurrent paths, within this
server's own noise.

## 3. Findings the report misses or gets wrong

1. **The capacity statement is overstated; the CPU tier is inclusive.** Every computed chunk is written through
   to host RAM, so data on the GPU also sits on the CPU. Usable capacity is therefore about the CPU size, not GPU
   plus CPU. Measured footprint: 80.1 GB stored for a 6.04M-token working set, i.e. **13.3 KB/token** including
   GDN checkpoints, against the README's 11.2 KB (attention only). 96 GiB (103 GB) therefore holds about **7.7M
   tokens, about 70 sessions of 110k in total**. The README says "~80 idle 110k-token sessions next to the GPU's
   own ~35" and "~9.2M tokens".
2. **The gain depends on sizing, and there is a cliff.**
   - Undersized tier, 12 GiB (`verify_results/load_bg.json`, 36 sessions plus probes): **0 external hits** in the
     whole run, with 241 GB written. A tier smaller than GPU KV holds nothing the GPU lacks, so it is pure cost.
   - 64 GiB with the clean seed-2 workload (working set about 1.2x the tier): mild degradation, as in the table
     in section 1.
   - 64 GiB with the same kind of 64-session load plus 16 probe sessions (`load64_bg.json`, about 1.45x the
     tier): **full collapse**, the same as no offloading. 61% of later turns lost their prefix, total hit rate
     was 20%, TTFT p50 198 s, 14 output tok/s. This run is not equal-workload with the baselines, because the
     probes add load; but the cause is visible: CPU LRU under cyclic session access thrashes, so stores doubled
     to 161 GB and there were only 1.4M external hits.

   So "one flag fixes it" holds only when the CPU tier is at least the total live working set (about 13.3
   KB/token). Past about 1.3-1.5x, the benefit disappears rather than degrading gracefully.
3. **Divergent sessions get no reuse at all**, on the GPU or the CPU. In every test, C = A[:62%] + a new tail
   gave cached = 0 on the first send (6/6 sequential, 16/16 under load), even though A sat in host RAM. GDN state
   is stored only at prompt-end checkpoints (rd(L)-2128), so a prompt that branches inside an earlier prompt
   falls back to recompute unless an earlier turn's checkpoint lies before the branch point. This is safe (no
   wrong state) but means offloading does nothing for history edits, compaction or rewinds. The original tests
   only used prompts that extend the previous one.
4. **The mixed GPU-hit + CPU-extension path was not exercised by anyone**, me included. My attempt (step 6 of
   `verify_seq.py`) came out as pure CPU loads: 0 local hits, all external. That path is untested.
5. **The claimed hit boundary is confirmed:** the restored length is rd(L_prev, 2128) - 2128 in every case.
   Startup with the 96 GiB tier took 204-267 s in my runs (393 s once, with 64 GiB) against 144 s for the
   baseline. Host RAM: container RSS 102 GiB, and host available memory fell from 206 GB to 109 GB.

## 4. Not re-verified (no GPU time spent)

These are taken from the report's own files only: LMCache MP + MTP corruption, SimpleCPUOffloadConnector,
`--prefix-match-unit 16`, raw PCIe bandwidth, and the per-restore H2D timings. I did see that CPU-restore TTFT ≈
GPU-hit TTFT in `seq.json`: cpuB 572-1159 ms against gpuB 424-994 ms.

## Reproduce

```
./verify_queue.sh                     # H1 seq correctness, H2 12 GiB load, H3/H4 64-session native/base (seed 2)
./verify_queue2.sh                    # repeat test (near-tie check)
./verify_queue3.sh                    # 64 GiB + probes (collapse)
./verify_queue4.sh                    # 64 GiB clean cliff run; 96 GiB + 48 sessions + probes (concurrency correctness)
python3 analyze.py verify_results/w700s2_base.json verify_results/w700s2_native.json verify_results/w700s2_off64.json
```
(The queue scripts take the lock themselves; `verify_run.sh` is the per-hold server wrapper.)
