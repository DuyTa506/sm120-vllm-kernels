# Skinny BF16/FP8 on current scalar + next

CPU-prepared only. No warm-cache transition or serving result has been measured in this round. Historical old-stack wins are not current-stack evidence. GDN replay and dynamic MTP are deliberately absent from this composition.

## Prior integration failure and isolated fix

Historical `decode-roofline/VERIFY.md` observed BF16 flag1→0 loading the same AOT artifact, leaving some compiled skinny behavior active. `dense-fp8-gemm/VERIFY.md` observed flag0→1 leaving all80 layer GEMMs on CUTLASS while eager lm_head changed. Both flags were read outside tracing and absent from `envs.compile_factors()`.

The isolated `generated/` copies retain historical numerical kernels and dispatch predicates. Only environment wiring changes: register `VLLM_SKINNY_BF16_GEMM` and `VLLM_SM120_SKINNY_FP8` (both default0), and `VLLM_SM120_SKINNY_FP8_MAX_M` (default64), then consume registered env values. MAX_M must also be keyed because it changes the custom-op runtime decision. Original directories/runtime remain unchanged. Original envs/utils/CUTLASS source hashes match installed vLLM0.32 files; FP8 CUDA bytes are identical to the historical candidate.

CPU `check_compile_keys.py` loaded composed envs and verified five distinct hashes for off, BF16, FP8, both and FP8 MAX_M32; CUDA remained uninitialized. Evidence: `compile_keys_cpu.json`. This proves key construction, not actual warm compiled-model activity.

## Prepared warm-cache and serving tests

`run.sh TAG off|bf16|fp8|both` inherits current scalar+CONFIGnext/FI1/fixedMTP2 controls, owned-container cleanup, shared GPU lock, source archives/manifests and startup verification of actual V2 runner, flags, BF16 dispatcher and composed file hashes. `CACHE_FAMILY` is mandatory: all transition runs use the same root and identical sources; clearing caches or using separate roots would hide the historical bug.

Root grant required; no command has been run:

```bash
sg docker -c 'bash kernels/decode-roofline/round2/warm_sequence.sh skinny_bf16_round2 bf16'
sg docker -c 'bash kernels/decode-roofline/round2/warm_sequence.sh skinny_fp8_round2 fp8'
```

Each sequence executes off→on→off→on with the same cache family. Each server measures c1/c4,32 requests per case,128 input/512 output, fixed seed, greedy sampling and detailed vLLM metrics. These random-token tasks match historical decode stress shapes; acceptance/quality generalization needs the real-text task suite before promotion. Startup/cold compilation is excluded from steady-state measurements. `check_warm_sequence.py` requires four successful terminals, same cache root, repeatable per-profile factor hashes, distinct off/on hashes and no failed benchmark requests.

**Remaining activation gate:** startup branch selection and configuration hashes do not prove the captured graph contains the intended kernels. Record CUDA-node traces for c1/c4 on the warm returns, as in existing `verify_insitu.py`/`verify_flag_off.sh`, and match actual kernel activity: skinny BF16 absent off/present eligible on; FP8 skinny absent off/present eligible on. Do not infer activation from timing or an eager lm_head alone. If startup profile guards pass but graph traces disagree, reject cache integration and investigate. The isolated runner now wires these checks with default TRACE=1: `serve_trace.sh` records CUDA graph nodes; after the standard requests, it captures short c1/c4 windows, exports SQLite and runs `check_trace.py`. Off requires no skinny kernel execution at either concurrency; enabled c1 requires captured-graph skinny activity, so eager lm_head alone cannot pass. C4 kernel counts are recorded without assuming M16. CPU synthetic-SQLite tests reject eager-only activation and disabled-kernel execution. GPU trace/export behavior remains untested; sources are ready for root review, not an empirically validated fix.

After cache correctness: paired AB/BA c1/c4 runs with identical prompts and current scalar+next, isolated BF16 and FP8 first, then stack if both survive. Record actual target/draft M buckets; historical c4 used M18, not the assumed16. Include task-quality and representative prefill no-regression checks before combined agentic evaluation.

## Strongest operand-matched baseline for new DeepGEMM QKVZ

Existing skinny FP8 uses per-token A scales, per-channel weight scales and BF16 output with epilogue `bf16_rn(sa * (sb * acc))`; FP32 K reduction order can differ from CUTLASS. It is an exact-operand comparator, not guaranteed bitwise-equal output.

For GDN QKVZ N12288/K2048, `_pick` selects(kw1,kb4) at M≤16 and(kw2,kb4) at M17..64. M65..128 falls back to CUTLASS. New DG fused QKVZ gains against CUTLASS at M32..64 therefore need comparison against this existing skinny candidate before dispatch selection. For M65/128 CUTLASS remains its fallback. Compare identical captured activation bytes, A scale values, prepared weights/B scales and full equivalent producer/epilogue costs; do not substitute a differently quantized block-scale operand or compare GEMM-only against a fused chain. The DeepGEMM agent owns this comparator and CPU extension build; this directory does not duplicate it or assume skinny is enabled in current CONFIGnext.

BF16 targets different operators: M≤3 for small N up to512, or M≤16/N≤128, principally router/small GDN projections and eligible draft calls. It is mostly disjoint from dense FP8 QKVZ, but changed router rounding can affect near-tied top-k choices, so current-stack numerical/task gates remain necessary.

After trace-verified warm transitions, use TRACE=0 for uninstrumented AB/BA throughput measurements with the same configurations. No claim of speed or completed warm-cache validation is made until those actual runs and source/activity checks pass. DeepGEMM agent built a separate exact original skinny comparator under `kernels/deepgemm-sm120/skinny_comparator/build`; this serving composition retains its original loader pending runtime-compatible artifact review.
