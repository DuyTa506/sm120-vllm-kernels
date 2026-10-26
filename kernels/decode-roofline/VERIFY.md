# Independent verification: decode-roofline / `VLLM_SKINNY_BF16_GEMM`

Verifier files (nothing of the author's was modified):
`verify_skinny.py` (kernel correctness and timing through the registered custom op), `verify_ab.sh`
(untraced end-to-end A/B), `verify_flag_off.sh` (checks the "default off" claim on a warm compile cache),
and `results/verify_*.txt` / `results/verify_flag*_server.log`.

## Verdict: PARTIALLY CONFIRMED

The speedup is real and reproduces: about -0.6 ms per decode step at c=1, smaller at c=2, and none at
c>=4. Numerics match cuBLAS to bf16 rounding. The patch is **not ready to integrate as written**: its
on/off switch is not part of vLLM's compile-cache key (section 4), and the end-to-end quality evidence
is only a kernel-level argument, because the greedy probe cannot resolve differences on this stack.

## 1. Kernel microbenchmark: reproduced

* `bench_skinny.py` (the author's script), re-run: identical within 0.05 us. Router 256x2048 at M=3 is
  7.73 -> 3.14 us, in_proj_ba 64x2048 at M=3 is 7.64 -> 2.39 us, with the same error figures.
* Methodology flaw, though not in the patch's favour: the bench says that 20 distinct weight copies
  "come from DRAM". They do not: 20 x 1 MB fits easily in the 128 MB L2. My bench cycles
  384 router copies (192 MB) or 1536 in_proj_ba copies, in a CUDA graph of 256 calls, median of 30,
  run in both orders. That bench makes cuBLAS look *worse*, and it matches the in-graph nsys number:

| shape | M | cuBLAS us (DRAM) | op us (DRAM) | path |
|---|---:|---:|---:|---|
| router 256x2048 | 1 | 2.38 | 2.36 | skinny |
| router 256x2048 | 2 / 3 | 11.50 / 11.50 | 2.57 / 3.37 | skinny |
| router 256x2048 | 6 / 12 | 3.96 / 4.06 | same (fallback) | F.linear |
| in_proj_ba 64x2048 | 1 / 2 / 3 | 2.16 / 7.28 / 7.29 | 1.25 / 1.55 / 2.01 | skinny |
| in_proj_ba 64x2048 | 6 / 12 | 7.30 / 7.52 | 2.19 / 2.33 | skinny |

  Estimate at M=3 (c=1 verify): 40 x 8.1 + 30 x 5.3 = ~0.48 ms per target forward. That is consistent
  with the -583 us measured on the target graph (the MTP router at M=1..3 adds a little more).
* The fallback rows are timed equal, so the gate adds no measurable cost when it is not taken.

## 2. Correctness: confirmed (kernel level)

`verify_skinny.py` goes through `torch.ops.vllm.skinny_bf16_linear`, using the patched `utils.py`
mounted over the image. It covers all 40 router gates, the MTP router and all 30 merged in_proj_ba
weights from the real checkpoint. Inputs are RMSNorm outputs built with the real layernorm weights and
outlier channels. Rows run at M = 1, 2, 3, 4, 6, 9, 12, 15 and 16, 1536 rows per weight.

* max |y - fp32 ref|: the op's error equals cuBLAS's error at every M (for example 3.10e-2 vs 3.10e-2
  at M=3). Max relative error is 3.5e-3 where the skinny path is taken and up to 5.9e-3 where it falls
  back (that is cuBLAS's own error). Outputs are 99.87-100% bitwise equal to cuBLAS.
* **Router top-8 expert sets** at M=3 differ from cuBLAS in 8 of 62,976 rows. The noise floor is the
  same: rounding the fp32 reference to bf16 also flips 8 of 62,976 against cuBLAS.
* in_proj_ba downstream: beta = sigmoid(b) error 1.5e-3, the same as cuBLAS. For g, the op's error
  (2.5e-3) is no larger than cuBLAS's (4.4e-2 worst case).
* Edge cases all pass: M=0, M=17 (falls back and is bitwise equal to F.linear), non-unit last-dim stride,
  row-strided weight view, bias, 3-D input, and odd N/K (16, 100, 512, K=1000, K=2056, K=4096).
* The op is bitwise repeatable. CUDA-graph capture and replay with new data equals eager.
  `torch.compile(dynamic=True, fullgraph=True)` around the op matches eager for M from 1 to 300.
* End to end: I re-ran `compare_probe.py` on the author's probe files and got the same figures
  (base vs base 425/1393 shared-prefix tokens, base vs skinny 308/1507 and 472/1395). The probe cannot
  tell the two apart, because the MTP-2 stack is not deterministic run to run. **No end-to-end quality
  metric (NLL or task eval) was run by the author or by me.** Given the bitwise-equality rates above I
  judge the risk low, but it is unmeasured.

## 3. End-to-end decode: confirmed at c=1, small at c=2, none at c>=4

`verify_ab.sh` gives a fresh server per side, CONFIG=optimized, CUDA graphs on, untraced, with
`vllm bench serve` random 128 in / 512 out. My runs are at 11:00 (skinny) and 12:45 (base) UTC.

| | base (mine) | skinny (mine) | author base / skinny |
|---|---|---|---|
| greedy c=1, median ITL, 3 seeds | 6.36 / 6.39 / 6.37 ms | 5.72 / 5.75 / 5.76 ms | 6.36 / 5.77 |
| greedy c=1, tok/s | 415 / 414 / 401 | 446 / 458 / 446 | 401 / 452 |
| greedy c=1, acceptance length | 2.80 / 2.78 / 2.69 | 2.74 / 2.81 / 2.74 | 2.70 / 2.78 |
| sampling c=1, median ITL, 2 seeds | 6.53 / 6.55 ms | 5.98 / 5.96 ms | 6.57 / 5.97 |
| greedy c=2 (M=6, in_proj_ba only), ITL | 6.57 ms | 6.41 ms | not tested |
| greedy c=4, ITL | 7.97 ms | 7.96 ms | 7.96 / 8.00 |

* c=1: ITL drops by **0.63 ms (-9.9%)** greedy and **0.57 ms (-8.7%)** sampled. The seed-to-seed spread
  is 0.03 ms, so this is far outside noise. Mean greedy tok/s rises 410 -> 450 (+9.8%) at an unchanged
  acceptance length. **The author's -0.6 ms per step holds.**
* c=2: -0.16 ms (-2.4%), from one run per side. Only in_proj_ba qualifies there, since the router at M=6
  falls back to cuBLAS.
* c>=4: no gain, as the author says. Prefill is not measured by anyone. By construction it takes the
  F.linear fallback at M>16, the same cuBLAS call through an extra opaque custom op, so a gain or loss
  there is not expected. Not verified.
* The author's own `results/{base,skinny}_decode.txt` (a server running under `nsys launch`) shows only
  -0.2 ms (6.57 -> 6.37). That data point was not explained in the README. My untraced runs agree with
  the -0.6 ms instead.

## 4. Integration hazards (the reason for "partially")

1. **The flag is not in the compile-cache key.** `VLLM_SKINNY_BF16_GEMM` is read with `os.environ`, so it
   is not one of vLLM's known env vars and `envs.compile_factors()` does not hash it. The torch.compile
   code hash covers the *source* of traced files only. The choice between `skinny_unquantized_gemm` and
   `default_unquantized_gemm` is made at layer init (`dispatch_unquantized_gemm`), not in traced code.
   v0.32 uses AOT compile (`torch_aot_compile/<key>`), and the key is the same for every variant (the log
   shows path `683d4d9d...` for both base and skinny). `_verify_source_unchanged` only compares file
   contents, and `evaluate_guards=False`. So once the patched `utils.py` is installed, flag=0 and flag=1
   **share one AOT artifact**, and whichever was compiled last is what runs. "Default off" is therefore
   not guaranteed on a warm `~/.cache/vllm`. Result of `verify_flag_off.sh`: see section 6.
   Fixes: add the flag to `vllm/envs.py` so it enters `compile_factors()`, or check the flag inside
   `skinny_bf16_linear_impl` (cheap; that code runs on the host only outside CUDA-graph replay).
2. **Cache thrash on the shared cache dir.** Base and patched servers alternating on the same
   `.cache/vllm` mount overwrite each other's AOT artifact. My skinny run logged "Compiling model again
   ... Source code has changed" for both backbone and eagle_head. The cost is extra startup time, not
   correctness, but other agents share the same cache dir.
3. **The patch ships a full-file copy of `layers/utils.py`** (mounted with `:ro`). The kv-offload area also
   carries its own `vllm/model_executor/layers/utils.py`. Integrating both would need a 3-way merge. The
   unified diff in `patch/` is small and applies to v0.32.0 (`orig/` matches the image byte for byte).
4. Thresholds (M<=3 for N<=512, M<=16 for N<=128) are tuned for SM120 only. The flag is not
   arch-gated, so turning it on elsewhere (SM90 or SM100 use GateLinear tiers 1-5 for the router anyway)
   is untested.
5. During my first flag-off attempt (`verify_ab.sh patched_off`) the engine failed to start, about 18 s
   after launch. The root cause was cut off because only `docker logs --tail 30` was kept. The rerun in section 6
   saved the full log and started cleanly.

## 5. Other claims (analysis, not code)

* `bytes_model.py` re-runs on the host and reproduces 4912 MB (2.73 ms) at c=1 and 40.8 GB (22.7 ms) at
  c=64. `decode_analyze.py` on the author's graph-level traces reproduces 4875.4 vs 4292.2 us target
  graph, 276 vs 287 steps, and busy 6515 vs 5884 us. `step_categories.py` on the node trace reproduces
  808 us across 70 BF16 small GEMMs (11.5 us each) and 1031 kernels per step.
* I did not independently verify the GDN "1R+3W" state-traffic claim or the 5 ms c=64 estimate for the
  unbuilt replay scheme. Both are estimates, presented as such.

## 6. Flag-off test (`verify_flag_off.sh`)

The test ran in one lock hold at 15:20-15:27 UTC, with the patched `utils.py` and `skinny_bf16_gemm.py`
mounted for both servers and the shared `.cache/vllm`. Output is in `results/verify_flag_off.txt` and
`results/verify_flag{1,0}_server.log`.

| server | compile-cache behaviour (from the log) | greedy c=1 ITL | tok/s |
|---|---|---:|---:|
| flag=1 | "Compiling model again ... Source code has changed" (backbone and eagle_head), then saved the AOT artifact | 5.80 ms | 435 |
| flag=0, started next | **"Directly load AOT compilation"** from the same `torch_aot_compile/683d4d9d...` and `a80e440d...` paths that flag=1 had just saved | **6.09 ms** | 427 |
| reference: base, unpatched (section 3) | | 6.36-6.39 ms | 401-415 |

The flag=0 server reused the artifact compiled with the flag on; the log shows this directly. Its ITL of
6.09 ms sits between "off" (6.37) and "on" (5.75), so part of the skinny path stayed active even though
the flag was 0. My reading is that the compiled piecewise graphs kept the custom op, while any linear
called outside the compiled graph re-dispatched at init. I did not verify that split. The 6.09 ms figure
is a single run, but the cache-load line does not depend on run-to-run noise. **Conclusion: with the
patch installed, `VLLM_SKINNY_BF16_GEMM=0` does not reliably turn it off on a warm compile cache, and
`=1` would likewise not reliably turn it on after a flag-0 compile.** This must be fixed before
integration, either by registering the env var in `vllm/envs.py` so it enters the compile hash, or by
re-checking the flag inside `skinny_bf16_linear_impl`. Clearing `~/.cache/vllm/torch_compile_cache` is the
operational workaround.

The engine crash in my first flag-off attempt (section 4, item 5) did not recur. The flag=0 server log
has no errors, so I treat that crash as transient and unexplained, not as a patch defect.

## 7. Summary for the orchestrator

* The claimed -0.6 ms per decode step at c=1 is **confirmed** (my runs: -0.63 ms greedy, -0.57 ms
  sampled, about +10% tok/s). c=2 gives -0.16 ms, c>=4 gives 0, and prefill is unchanged by construction
  but not measured.
* Numerics are **confirmed** equal to cuBLAS within bf16 rounding on all 71 real weights. Router top-8
  flips are at the noise floor. No end-to-end quality eval was run by anyone.
* **Integration blocker:** the env flag is not in the compile-cache key, which I demonstrated with an AOT
  artifact reused across flag values. Small fix, listed above.
* Minor: the author's bench claims its reads come from DRAM, but they are L2-resident. Correcting this
  favours the patch: cuBLAS takes 11.5 us from DRAM at M=2-3.
