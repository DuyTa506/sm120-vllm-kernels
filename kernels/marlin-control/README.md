# Same-checkpoint Marlin control: CPU preparation

Purpose: test whether retaining FP16/BF16 activations avoids enough native FP4 quantization/routing overhead to improve the current Qwen workload. Keep `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast`, compressed-tensors metadata, CONFIGnext attention, scalar parent sources and fixed MTP2. Only target MoE backend changes `flashinfer_b12x`→`marlin`; explicitly keep draft `moe_backend=flashinfer_b12x` for both controls. Do not force ModelOpt or substitute NVIDIA/RedHat/GPTQ weights.

This is W4A4 versus W4A16 execution of the same checkpoint, not a numerical-equivalence claim. The current vLLM oracle provides the mechanism: select Marlin for NVFP4, repack existing E2M1 weights and scales, and construct the A16 quantization config. CPU import/source audit verifies the class mapping and NVFP4-weight/A16 scheme, while full deployment eligibility and GPU weight preparation remain untested.

## Reproduce preparation

From the qwen-prefill worktree:

```bash
python3 kernels/marlin-control/prepare_probe.py --output kernels/marlin-control/results/plan_run1
```

The preparer verifies a frozen stack `parent_ready2` source bundle, reads the captured checkpoint metadata and CPU runtime audit, and writes separate Docker argv plus a provenance manifest. It checks that after output/container-name normalization the **only command difference is target MoE backend**. It does not execute Docker/GPU. `results/runtime_source_audit.json` contains exact installed source texts/hashes and CPU class-selection evidence; CUDA was hidden and uninitialized. Refresh that audit in the chosen image if its identity differs; do not silently trust a tag that changed.

## Required staged harness

1. **CPU eligibility/contract:** retain full target and draft runtime source hashes, checkpoint config/index and quantization descriptors. Before GPU, validate Marlin's complete deployment support for our expert count/dtype/activation and MTP configuration; class-level scheme support alone is insufficient. Confirm actual target routes to `MarlinExperts` and draft remains the BF16-compatible B12x implementation after construction. Keep unsupported results rather than rewriting checkpoint metadata.
2. **Small numerical probe:** one root-granted server/control, fresh output/cache and shared GPU lease, no automatic next job. Read back class/backend, packed weight shapes/dtypes/scales and checkpoint identity. For captured target layer0/19/39, compare identical packed weight semantics, token rows and router choices at M1/3/16/64/1024/4096. Compare Marlin against BF16/FP32 dequantized reference and native B12x separately; include global scales and complete two-GEMM/SwiGLU/routing/reduction. Record changed activation quantization as part of the backend effect. Changed-input CUDA graph replay must update activations and detect stale captured storage. Do not use W4A4 bit equality as the Marlin acceptance criterion.
3. **Quality screen:** matched real prompts, repeated cold NLL in each variant (at least five rounds), exact prompt/token hashes and raw per-token differences. Baseline self-noise is already substantial; an average improvement does not prove equivalence. Investigate discrepancies before performance promotion. Fixed reference/quality criteria require root review.
4. **Focused timing:** same six repository prefill lengths and four greedy decode cases, temperature0 and fixed seeds; report prefill/ITL separately plus full GPU graph critical path. Include quantize/route/reduce costs, not isolated GEMM only. For modest differences, repeat in reversed process order; inspect fresh-cache selected configs. Compare W4A4 versus W4A16 as a backend package effect.
5. **Composition last:** only if useful, compose with supported patches and run final agentic benchmark once. Marlin does not use the native dynamic expert scheduler, so native long-first and Marlin are alternative target-MoE profiles, not additive gains. Attention and independently eligible dense patches may still compose.

No server lifecycle runner/GPU benchmark is claimed complete by this preparation. The saved argv are reviewable plans; root must supply/approve the isolated lifecycle and selection/readback harness before execution. Humming, replay-state patches and speculative-depth changes are excluded.
