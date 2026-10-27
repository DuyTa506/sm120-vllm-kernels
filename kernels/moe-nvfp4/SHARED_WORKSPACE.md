# Qwen prefill: shared b12x workspace

This candidate preserves the scale-corrected hybrid backend's kernels and
shares their scratch across sequential MoE layers. It targets
`unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` on one RTX PRO 6000 Blackwell, TP=1/EP=1,
vLLM v0.32.0, CONFIG=next and MTP-2. It is opt-in. Clean single-request
prefill A/B and the 17-scenario serving task matrix completed. Decode results
are mixed; exact-alpha and explicit-greedy follow-up remain pending.

## What changes

`VLLM_B12X_SHARED_WORKSPACE=1` selects shared static/dynamic workspace and
temporary output through vLLM's WorkspaceManager. The manager partitions
resources by execution lane and ubatch. Every layer keeps a separate wrapper
for cached weight views and folded scales. `output.copy_` preserves ownership
of returned results. Small batches return through CUTLASS before allocating
any b12x wrapper.

Apply `04-moe-b12x-shared-workspace.patch` **after** patch 03, or mount the
candidate `shared/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py`
over the same vLLM path along with patch 03's unquantized-MTP fallback. Keep
`VLLM_B12X_FIX_SCALES=1` and `--moe-backend flashinfer_b12x`.

## Verified so far

GPU allocations use the actual FlashInfer wrappers; these are not mocked:

| Check | Legacy hybrid | Shared candidate |
|---|---:|---:|
| Extra scratch/output for a second identical layer | 579.652 MiB | 0.000 MiB |
| Total scratch/output for two layers | 1159.305 MiB | 579.652 MiB |

Eight allocation/lifetime tests pass, after demonstrating the expected failures
on the unchanged baseline. They cover owner/ubatch/configuration isolation,
manager lock/reset, default-off behavior, and decode bypass.

Real checkpoint layers 0 and 19 pass 120 output comparisons at 3, 12, 1023,
1024, 1025, 8192 and 16384 tokens. CUDA graph pointers stay fixed while routing
changes among real, eight hot experts, masked tails, all masked, and shifted
expert IDs. Tolerance is compared against repeated execution of the unchanged
b12x backend, which is itself nondeterministic. This establishes consistency
with that backend, not absolute full-model accuracy or bitwise equality.

At 8k, two-layer b12x graph time was 1319/1342 us before and 1312/1337 us after
in the two timing orders. This is effectively unchanged; sharing memory is
not itself a demonstrated GEMM speedup. The opportunity is to retain the
existing b12x advantage over CUTLASS without duplicating its large workspace.

## Serving measurement

Use `shared_workspace_e2e.sh <tag> <cutlass|shared>` in the docker group.
The runner uses CONFIG=next, separate compile caches, a private test server
on port 8132, and `--gpu-memory-utilization 0.60` to coexist with the unrelated
training job. Set `NSYS=1` for the prefill GPU-span capture.

The user authorized shared-GPU measurements. Compare both 16-request seeds
at 8k/16k; save individual TTFTs, decode c=1/c=4 and GPU telemetry. Warmup
prompts use disjoint seeds and measured calls disable the CLI readiness probe
and warmup, both of which otherwise reuse the first measured prompt. Prefix
cache reuse across lengths is avoided by including the input length in both
the measured and warmup seeds. Equal seeds produce the same token prefixes
at different lengths. The runner labels this protocol `cold-v2`, and also
warms decode separately before measuring it.

`results/shared_a1` is excluded from speedup conclusions: it used the older
warmup protocol and was interrupted by an in-flight runner edit. Corrected
A2/B1 results are exploratory: their 8k prompts are cold, but their 16k prompts
can reuse prefixes from 8k. A reset attempt returned HTTP 404; no cache reset
occurred. Their decode c=4 runs also include first-use compilation outliers.
Do not present the 16k timings as cold-prefill results or compare their decode
TTFTs as warmed serving latency.

### Completed shared-GPU run (A2 versus B1)

Both variants served all requested workloads while the unrelated Arm A
training process was active. These sequential runs do not establish causality:
the competing workload changed during the experiment.

| Measurement | CUTLASS A2 | Shared hybrid B1 |
|---|---:|---:|
| 8k mean TTFT, seed 101, 16 requests | 279.78 ms | 254.44 ms |
| 8k mean TTFT, seed 202, 16 requests | 259.11 ms | 236.18 ms |
| 8k GPU span, mean of four profiled requests | 222.55 ms | 202.93 ms |
| Decode median ITL, c=1 | 11.27 ms | 10.69 ms |
| Decode median ITL, c=4 | 13.96 ms | 13.00 ms |
| Reported KV cache tokens at utilization 0.60 | 2,078,181 | 2,145,441 |

The two 8k seed means are 9.06% and 8.85% lower respectively. This is a
promising signal, not a verified speedup under an idle GPU. Decode also sped
up despite preserving its CUTLASS path, and several unchanged kernel timings
vary substantially. Whole-run SM clock means were close (2679/2688 MHz), but
that does not control contention during individual requests.

The full candidate server successfully initialized all 40 layers and retained
KV capacity comparable to CUTLASS. Its slightly larger reported KV capacity
is not evidence that it consumes less memory than CUTLASS: vLLM's memory
profiling ran under changing external GPU load. The direct allocation tests
above establish the scratch reduction against **unshared b12x**, a different
baseline. CUDA graph traces, detailed request samples and telemetry are in
`results/shared_a2/`, `results/shared_b1/`, and `results/nsys/raw/shared_*_8k.*`.

### Clean GPU run (A3 versus B2, cold-v2)

Arm A training exited before these runs. Only the benchmark EngineCore was
listed as a GPU compute process. Same CONFIG=next, MTP-2, memory utilization
0.60, image, two 16-request seeds per length, separate compile caches.
An independent CPU reconstruction of all 112 prompts across 14 phases found
zero shared leading tokens, including warmups and profiling requests
(`results/cold_v2_prefix_audit.json`). The task-matrix runner additionally
uses explicit per-phase cache salts to prevent collisions as request counts grow.

| Measurement | CUTLASS A3 | Shared hybrid B2 | change |
|---|---:|---:|---:|
| 8k mean TTFT, seed 8293 | 198.52 ms | 177.77 ms | -10.46% |
| 8k mean TTFT, seed 8394 | 187.42 ms | 178.71 ms | -4.65% |
| 16k mean TTFT, seed 16485 | 391.35 ms | 359.87 ms | -8.04% |
| 16k mean TTFT, seed 16586 | 406.13 ms | 364.63 ms | -10.22% |
| 8k GPU span, four requests | 132.33 ms | 120.23 ms | -9.14% |
| Decode c=1 median ITL | 6.53 ms | 6.48 ms | -0.8% |
| Decode c=4 median ITL | 8.13 ms | 8.11 ms | -0.2% |
| Decode c=1 output tokens/s | 300.43 | 281.35 | -6.35% |
| Decode c=4 output tokens/s | 789.77 | 739.53 | -6.36% |
| Reported KV tokens | 2,078,181 | 2,145,441 | +3.24% |

The baseline's four GPU spans were 131.9–132.6 ms and the candidate's
119.6–120.7 ms. This resolves a prefill saving in this clean sequential pair.
It is not a claim of a uniform 9% improvement across serving workloads.
Decode ITL is similar, but output throughput is lower and small-prompt TTFT
increases (c=1 mean 48.84→60.66 ms, c=4 66.61→105.05 ms). The later reversed-order task matrix below checks this result with wider
workloads and recorded MTP statistics.

Clean real-routing layer-19 microbenchmark, including ID masks and output copy:

| tokens | CUTLASS us | b12x us | reduction |
|---:|---:|---:|---:|
| 1024 | 321.8 | 318.9 | 0.9% |
| 2048 | 386.0 | 341.8 | 11.5% |
| 4096 | 553.5 | 447.7 | 19.1% |
| 8192 | 887.0 | 654.6 | 26.2% |
| 16384 | 1567.7 | 1208.8 | 22.9% |
| 8192 balanced routing | 884.0 | 651.6 | 26.3% |

Both interleaved timing orders agree. At 8k this is 232.4 us/layer, about
9.3 ms across 40 layers, consistent in scale with the in-server span saving.
Do not attribute all 12.1 ms of server difference to MoE without a kernel-node
trace. Raw results: `results/b12x_clean_micro_L19.txt`, `results/shared_a3/`,
`results/shared_b2/`.

A clean shared-versus-unshared replay additionally passed ten comparisons
and retained the allocation reduction. Two-layer times were 1284/1293 us
unshared and 1277/1285 us shared in opposite timing orders; sharing itself
remains primarily a memory improvement (`results/shared_workspace_clean.jsonl`).

## Full serving task matrix (shared then next, utilization 0.78)

Both runs completed all 17 workloads and their quality probes. The full
[paired table](../../results/qwen_sequential/shared_vs_next_tasks_1.serving.md)
retains TTFT, ITL, throughput and MTP acceptance per scenario.

| cold prefill | next TTFT p50 ms | shared TTFT p50 ms | change |
|---|---:|---:|---:|
| 2048 | 66.2 | 65.7 | -0.8% |
| 4096 | 106.9 | 98.8 | -7.6% |
| 5000 | 122.6 | 116.2 | -5.2% |
| 8192 | 191.6 | 164.3 | -14.3% |
| 16384 | 384.5 | 329.5 | -14.3% |
| 32768 | 866.3 | 756.6 | -12.7% |

Seven mixed serving workloads improve total throughput by +0.2% to +5.2%
in this single pair. TTFT does not improve uniformly: prefill-heavy, RAG and
shared-prefix medians increase by 5.9%, 8.6% and 3.2%. Decode output rates
at c1/4/16/64 change +10.3%/-3.0%/+2.2%/-1.7%; median ITL remains close.
The earlier -6.35% decode-rate difference does not reproduce uniformly.
These requests inherit checkpoint sampling temperature 1.0/top-k 20/top-p .95;
dataset seeds alone do not fix generations or MTP acceptance. For example,
c1 acceptance changes 44.1% to 53.5%. This is no demonstrated decode-kernel
speedup. A temperature-zero diagnostic is prepared separately.

Both variants pass 20/20 needles. NLL changes differ in sign by corpus;
checkpoint-distribution agreement also differs. These limited probes do not
establish identical model behavior. Exact-alpha follow-up is documented in
[CUTLASS_SCALE_ROUNDING.md](CUTLASS_SCALE_ROUNDING.md); its GPU tests pass,
but the serving results above used the original `shared` overlay.
No agentic benchmark has run for this individual patch.

## Reproduction and scope

Allocation tests in the vLLM image, from this directory:

```sh
python3 -m pytest test_shared_workspace.py -q -s
python3 verify_shared_workspace.py --tokens 1023 1024 1025 8192 --replays 10
python3 verify_shared_workspace.py --tokens 3 12 16384 --replays 5
```

The pinned local image ID is
`sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34`.
It exposes `WorkspaceManager.get_persistent_resource` and FlashInfer's three
`shared_*` constructor arguments. The candidate explicitly requires the
persistent-resource API; it does not silently fall back to per-layer buffers.

Calls sharing an owner must be serialized; overlapping forward streams need
separate owners. EP, other output dtypes, sleep-mode reclamation and changing
model dimensions at runtime have not been validated. The approximately
48 MiB/layer represented by MMA-layout scale tensors aliases existing scales:
the installed conversion is `view` plus `permute`, not an additional allocation.
The old verifier's theoretical 24.52 GiB total double-counted those scales;
40 unshared scratch/output allocations total 22.64 GiB. The directly measured
579.652 MiB per wrapper and the sharing reduction remain valid. Keep this
candidate optional until serving results support adoption.
