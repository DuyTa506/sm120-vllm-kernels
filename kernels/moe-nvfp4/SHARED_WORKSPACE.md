# Qwen prefill: shared b12x workspace

This candidate preserves the scale-corrected hybrid backend's kernels and
shares their scratch across sequential MoE layers. It targets
`unsloth/Qwen3.6-35B-A3B-NVFP4-Fast` on one RTX PRO 6000 Blackwell, TP=1/EP=1,
vLLM v0.32.0, CONFIG=next and MTP-2. It is opt-in; initial serving validation
passes, but performance measurements without competing GPU work are pending.

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

Real checkpoint layers 0 and 19 pass 110 output comparisons at 3, 12, 1023,
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
48 MiB/layer of transformed weight scales remains; this change shares scratch,
not weights. Keep this candidate optional until serving results support adoption.
