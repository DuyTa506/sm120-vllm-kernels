# Experimental round2 composition

Current revision: **ready2**. Ready1 is superseded and must not be launched: its class-only substitution did not qualify FlashInfer’s persisted native cache key. Ready2 uses the exact source-hash tuple suffix verified in task-order serving b3; see `results/runtime_cpu_ready2.json`. This stack revision itself has not run on GPU.

CPU-prepared overlay for Qwen3.6-35B-A3B-NVFP4-Fast on vLLM 0.32 / SM120. **No composite GPU correctness, performance or quality result yet.** This is a reproducible experiment, not a promoted production configuration. The existing parent is CONFIG=next + scalar NVFP4 MoE + fixed MTP2. All four profiles use the same dev image and fixed DeepGEMM PDL=false policy for matched comparison.

| Profile | DG rowwise FP8 | Skinny FP8 | Skinny BF16 | Native long-first MoE |
|---|---:|---:|---:|---:|
| parent (default) | 0 | 0 | 0 | 0 |
| task_order | 0 | 0 | 0 | 1 |
| dense | 1 | 1 | 1 | 0 |
| combined | 1 | 1 | 1 | 1 |

The builder copies the already composed DG scaled_mm sources and adds the existing skinny BF16 utils/kernel. It does not splice competing dispatch branches. For eligible QKVZ K2048/N12288, DG owns only M={32,33,48,63,64,65,80,96,112,128}; otherwise the existing skinny selection runs, with CUTLASS fallback. The original operand/layout/dtype/scale/bias guards remain intact. At requested boundary M1/16 the dense profile chooses skinny; M32/64/65/128 DG; M129/1024/4096 CUTLASS. Therefore dense changes alone do not accelerate the large-M prefill GEMM; native task ordering is the prefill-oriented candidate. BF16 ownership is separate: small unquantized linears, M≤3 for N≤512 and M≤16 for N≤128.

The native plugin directly substitutes the native kernel class after checking baseline and candidate source hashes. It has no worker warmup patches, constructor wrappers, workspace collection, NVTX ranges, queue reads or benchmark RPC methods. Long-first changes task order, not model routing. Before any native cache population, both OFF and ON qualify `_dynamic_kernel_cache_key` with `(task_order_source_sha256, selected_implementation_sha)`, retaining the complete original tuple. This separates both native in-memory and persisted disk artifacts; static/micro cache keys are unchanged. A populated dynamic cache or profile/flag mismatch fails closed. The source-qualified mechanism produced 40 actual LongFirst live4096 graph nodes in the separate task-order b3 run, which validates activation there, not this combined stack or a performance gain. `VLLM_SM120_TASK_ORDER_LONG_FIRST` is a compile factor. All dense, parent patch flags, skinny max-M, source identity and explicit Torch Inductor PDL participate in the merged env registry and compile factors.

DeepGEMM PDL is fixed false before model load. `--inductor-pdl 0|1` independently controls TORCHINDUCTOR_ENABLE_PDL; default0. PDL1 is not implied by any composite profile and has no accepted performance/numerical verdict here.

## Prepare and inspect

Run from the qwen-prefill worktree. No GPU or package installation is used:

```bash
python3 kernels/stack-round2/prepare.py --output kernels/stack-round2/bundles/parent_run1
python3 kernels/stack-round2/prepare.py --profile combined --output kernels/stack-round2/bundles/combined_run1
python3 kernels/stack-round2/cpu_check.py kernels/stack-round2/bundles/combined_run1
python3 kernels/stack-round2/serve.py --bundle kernels/stack-round2/bundles/combined_run1 \
  --output kernels/stack-round2/results/combined_run1 --name stack-round2-combined-run1
```

The last command only prints Docker argv. `--execute` starts the foreground owned server, holding the shared GPU lock for its entire lifetime; only run with root's GPU lease. No benchmark or agentic suite runs automatically. The launcher refuses occupied GPUs, existing containers and reused output directories; there is no forced removal of a pre-existing container. It resolves the Docker image to an immutable image ID, writes argv/manifest/logs, and cleans up its owned container on termination, records vacancy and rechecks source hashes. FI shutdown verification is separate and can fail the run; a clean container alone is not evidence of a valid benchmark.

```bash
sg docker -c 'python3 kernels/stack-round2/serve.py --bundle kernels/stack-round2/bundles/combined_run1 --output kernels/stack-round2/results/combined_run1 --name stack-round2-combined-run1 --execute'
```

Endpoint is localhost:8140; port/GPU/HF-cache path are explicit CLI options. Existing repository serving/eval clients can use this endpoint. The launcher uses full next capture sizes/max-batched-tokens16384, not the smaller diagnostic DG screen. Thus first GPU startup can exercise combinations not previously validated. Root reviews before GPU smoke; benchmark attribution requires actual participating kernels, repeated quality controls and matched workload runs.

## Source and cache isolation

Each fresh bundle stores copied source files plus hashes of every preparation input, output and all 6,945 read-only external DG source/binary files. Source ID incorporates profile/flags/full manifest. The launcher rejects changed inputs, bundle outputs, manifest identity, DG file set or file contents. It mounts files individually read-only, copies fixed FI controls/reference into the bundle and preserves the scalar oracle/experts sources. Builder, launcher and CPU checker are included as bundle provenance. No shared scripts, generated sources, installed packages, PDL frozen runner or other agents' directories are changed.

Each run has distinct output and VLLM/Inductor/Triton/DG/extensions/FlashInfer cache paths using source ID and explicit Inductor PDL mode. Fresh cache isolation does not pin autotuned algorithms: any future performance/quality attribution must audit chosen configs or use a controlled shared-config protocol.

## CPU evidence and outstanding gates

`cpu_check.py` executes the actual AST dispatcher bodies with mocked GPU calls: four DG/skinny flag combinations at nine requested M boundaries, one final FP8 owner per case, all-off fallback, misalignment fallback and 30 BF16 eligibility boundaries. This does not establish tensor numerics. CUDA-hidden real-runtime checks exercise the generated env module's actual compile_factors(), toggle every registered composite/parent flag, import the native baseline/candidate and verify direct class replacement while torch CUDA remains uninitialized.

`parent_v1`, `combined_v2`, `combined_v3`, and every `*_ready1` bundle are development provenance, superseded by `*_ready2` bundles and reports. Do not launch these earlier bundles; later source changes deliberately make their input-integrity check fail.

Pending, deliberately absent: Humming per-shape adapter; GDN replay until the preemption gate; dynamic MTP. Also absent: losing rowwise swap/tile64 variants and speculative new policies. The experimental composed stack does not mean every patch is beneficial. Agentic evaluation remains the final composed-config task after kernel activation, numerical/quality investigation and prefill/decode checks.
