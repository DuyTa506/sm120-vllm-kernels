# Verification of the moe-nvfp4 claim (b12x scale fix + hybrid dispatch)

**2026-10-28 addendum:** this report describes the original unshared backend.
The [shared-workspace follow-up](SHARED_WORKSPACE.md) removes its per-layer
scratch duplication and has a clean prefill A/B; full task validation is ongoing.
The theoretical 24.5 GiB figure below double-counted 48 MiB/layer of MMA scale
views as copies. Installed `convert_sf_to_mma_layout` is view+permute, so the
40 unshared scratch/output allocations total 22.64 GiB. The measured per-wrapper
579.65 MiB and original server KV figures remain valid; raw reports are retained.

Verifier notes, 2026-10-25/26. I did not modify the author's code. My files: `verify_bench.py`, `verify_numerics.py`,
`verify_logprobs.py`, `verify_e2e.sh`. Results: `results/verify_kernel.txt` and `results/verify_{hybrid,cutlass,hybrid_b,cutlass_b}.*`.

**Verdict: partially confirmed.**
- Confirmed: the diagnosis, the accuracy fix, and the kernel-level speedup.
- Not confirmed: the end-to-end TTFT gain. It is within the run-to-run noise of the shared GPU.
- New problem: the patch as shipped **cuts the KV cache almost in half**, from 3.37M to 1.77M tokens. The README
  understates this cost as "about 2 GB".

## 1. What holds up

| claim | my result |
|---|---|
| vLLM's b12x bake pushes the weight block scales into e4m3 subnormals | Reproduced on CPU with `check_b12x_scale_bake.py` for layers 0/19/39: 100 % subnormal, mean scale error 12-48 %, 36 % flushed to zero in layer 0 gate/up. The numbers are identical to the README. |
| The fixed b12x matches CUTLASS accuracy against fp32 | Reproduced at the full prefill size M=8192, which the author did not test (they used 4096). Layer 19: CUTLASS 0.1334 vs b12x 0.1334 rel-L2. Layer 39: 0.1001 vs 0.1002. b12x vs CUTLASS differ by 0.024 / 0.022 rel-L2; per-token p50 is 0.02, p99 0.05, max 0.10. The outputs are finite. |
| Around the hybrid threshold | At M = 1000, 1024, 1025, 1100, 1536 and 3000, b12x vs fp32 equals CUTLASS vs fp32 to within 0.0002. There are no edge-size failures. |
| Expert id -1 masking, zero (padding) rows | Both OK: outputs are finite, zero rows give zero output, and masked ids match CUTLASS (0.02 rel-L2). |
| Kernel speedup at 8192 tokens | Reproduced, and not specific to layer 19 or its skewed routing. The table below uses CUDA graphs, interleaved A/B replay in two passes with the order reversed, and a median of 40×5. |
| Patch hygiene | `03-*.patch` applies cleanly to the pristine v0.32.0 files, and the result is byte-identical to `patched/`. Everything is gated by `VLLM_B12X_FIX_SCALES=1` plus `--moe-backend flashinfer_b12x`, both off by default. The direct-micro (M≤3) zero-output bug can only be reached when the threshold is below 4, and the code then falls back to unit scales. That logic is correct. |
| Decode unchanged | Median ITL at c=1 was 6.54 / 6.44 ms (hybrid) vs 6.51 / 6.51 ms (cutlass). MTP acceptance was 63.6 % / 56.9 % (hybrid) vs 64.5 % / 61.8 % (cutlass); it is noisy, with no consistent sign. |

Kernel time per MoE call in us, CUDA graph, real routing unless noted. `b12x_int` adds what the patched `apply()`
really does on top of the author's bench: the negative-id mask (compare + 2× masked_fill), `.to(int32)`, and
`output.copy_` of the 32 MB result. These cost **about 18-20 us per call at 8k**, which the author's bench leaves out.

| layer | M | CUTLASS | b12x_int | b12x_raw (author's method) | int/cut |
|---|---:|---:|---:|---:|---:|
| 19 | 1024 | 321.7 | 318.6 | 309.6 | 0.99 |
| 19 | 2048 | 387.7 | 343.5 | 332.2 | 0.89 |
| 19 | 4096 | 561.0 | 460.0 | 447.7 | 0.82 |
| 19 | 8192 | 903.1 | 677.4 | 659.3 | 0.75 |
| 19 | 16384 | 1623.2 | 1254.4 | 1208.1 | 0.77 |
| 19 | 8192 balanced random routing | 897.5 | 674.2 | 654.3 | 0.75 |
| 0 | 2048 | 412.1 | 404.5 | 393.2 | 0.98 |
| 0 | 8192 | 895.3 | 666.9 | 646.5 | 0.745 |
| 39 | 8192 | 904.8 | 685.0 | 664.5 | 0.757 |

- Saving per layer at 8192 tokens, including the integration overhead: 220-228 us. Over 40 layers that is about
  **9.0 ms of GPU time per 8k prefill**; the claim was 9.6 ms.
- Between 1024 and 2048 tokens the gain is only 1-11 %, and in layer 0 at 2048 it is 2 %.
- My absolute CUTLASS numbers are about 3 % higher than the author's (903 vs 871). The ratio is the same.

## 2. What does not hold up or was missed

1. **The KV cache is almost halved.** This is the main integration hazard, and it is unmeasured and understated in the README.
   - vLLM creates one `B12xMoEWrapper(use_cuda_graph=True, max_num_tokens=max_num_batched_tokens=16384)` per MoE
     layer.
   - Each wrapper pre-allocates a static workspace, a dynamic workspace and a 16384×2048 bf16 output buffer. I
     measured **580 MiB per layer** with `torch.cuda.memory_allocated`. Adding 48 MiB of MMA-layout scale copies,
     40 layers come to **24.5 GiB**.
   - Server logs, same CONFIG=optimized flags:

     | | available KV memory | KV cache tokens |
     |---|---:|---:|
     | cutlass | 44.62 GiB | 3,369,930 |
     | hybrid | 23.5 GiB | 1,774,645 (-47 %) |

     Both hybrid runs logged the same figures.
   - This hurts prefix-cache hit rate and high-concurrency throughput. The author lists the throughput sweep as not
     measured.
   - The fix is easy and should be required before integration. FlashInfer's wrapper accepts
     `shared_static_workspace` / `shared_dynamic_workspace` / `shared_output` exactly for the one-wrapper-per-layer case.
     The hybrid also never uses the static (small-M) b12x workspace.
   - My first hybrid server start failed in engine-core initialization. Another agent's server container was on the
     GPU at that moment. The root-cause lines were lost, so I cannot attribute the failure; the second and third
     starts succeeded.
2. **The end-to-end TTFT gain is not robustly reproduced.**
   - Setup: 8k random prompts, `vllm bench serve`, concurrency 1, 16 prompts per seed. The runs were done in A/B/A/B
     server order: hybrid, cutlass, hybrid_b, cutlass_b.

     | run | seed 101 | seed 202 | seed 303 | seed 404 | 16k mean |
     |---|---:|---:|---:|---:|---:|
     | cutlass | 191.4 | 187.4 | | | 354.5 |
     | hybrid | 191.6 | 189.2 | | | 362.7 |
     | cutlass_b | 186.7 | 192.5 | 194.0 | 189.5 | 329.7 |
     | hybrid_b | 191.4 | 182.1 | 178.3 | 181.7 | 299.2 |

   - Pooled 8k mean: cutlass 190.3 ms, hybrid 185.7 ms, a difference of -4.5 ms.
   - Pair A shows +1 ms and pair B shows -7 ms. Seed 101 shows no gain in either hybrid run. The 16k numbers swing by 60 ms between
     identical configs.
   - The author's single cutlass run (194.6 ms) against two hybrid runs (182.8 / 188.2 ms) sits inside the same
     noise band.
   - What the data supports: "prefill GPU time should drop by about 9 ms per 8k (kernel-level), and an end-to-end
     improvement of 0-7 ms is consistent with that but not resolved."
   - An nsys GPU-span A/B is still needed to confirm the in-server saving.
3. **Quality checks at the e2e level have no statistical power.** The server is strongly nondeterministic by itself:
   - Two **identical** hybrid runs give a mean |Δ logprob| of 0.11-0.21 per prompt token, with single-token
     differences up to 15 nats. Prompt NLL for a 3.4k-token prompt was 0.324 vs 0.405 between those two identical runs.
   - Cross-config differences, cutlass vs hybrid, are the same size.
   - So the NLL table in the README ("+0.002 to +0.004, within noise") cannot detect a regression of that order either
     way.
   - The trustworthy quality evidence is kernel-level: identical error against fp32 on 5 layers at M=4096, and on
     layers 19/39 at M=8192.
   - Needles: 20/20 in both of my full runs.
   - This nondeterminism comes from the optimized server stack in general, not from this patch. The CUTLASS MoE
     kernel itself is bitwise deterministic.
4. **New numerical properties of the b12x path (flag, not blocking):**
   - b12x is **not run-to-run deterministic**: the same inputs give max_abs differences of 2e-3 (layer 19) and 3e-2
     (layer 39). CUTLASS is bitwise reproducible.
   - b12x is **not batch-invariant**: the same token's output depends on the other tokens in the batch, with max_abs
     0.125 in layer 39 and 0.83 when padding rows are added. CUTLASS showed 0 difference.
   - The hybrid also switches kernels at 1024 tokens. The same tokens differ by about 2-2.6 % rel-L2 depending on
     whether the batch is 1023 or 1024 tokens. Prefix-cache chunk boundaries can therefore change numerics.
   - The magnitudes are within FP4 rounding noise, and the accuracy against fp32 is the same.

## 3. Bottom line

- The fix itself is correct and worth upstreaming: the vLLM bake bug and the FlashInfer scale-convention mismatch are
  real.
- The kernel speedup (≈0.75× at ≥4k tokens, ≈9 ms of GPU time per 8k prefill) is real.
- Integrating the patch as is trades **47 % of the KV cache** for a TTFT gain that I could not separate from noise
  (≈4.5 ± 4 ms on about 190 ms).
- Do not integrate until the wrappers share one workspace and output buffer across layers, and the KV-cache size and
  an nsys prefill span have been re-measured.

## Reproduce

```
cd /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels
# kernel speed + numerics (one ~6 min hold)
flock $LOCK sg docker -c "docker run --rm --name moe-nvfp4-verify-k --gpus device=0 --ipc=host --entrypoint bash \
  -e HF_HUB_OFFLINE=1 -v /opt/intramind/hf-cache:/root/.cache/huggingface:ro -v $PWD/.cache/flashinfer:/root/.cache/flashinfer \
  -v $PWD:/work -w /work/kernels/moe-nvfp4 vllm/vllm-openai:v0.32.0 \
  -c 'python3 verify_bench.py 19 0 39; python3 verify_numerics.py 19 39'"
# e2e (each takes the lock itself, ~5 min hold): KV size, logprobs, long_eval, 8k/16k TTFT, decode c=1
kernels/moe-nvfp4/verify_e2e.sh hybrid b12x_fixed; kernels/moe-nvfp4/verify_e2e.sh cutlass cutlass
TTFT_ONLY=1 N16=16 kernels/moe-nvfp4/verify_e2e.sh hybrid_b b12x_fixed; TTFT_ONLY=1 N16=16 kernels/moe-nvfp4/verify_e2e.sh cutlass_b cutlass
python3 -I kernels/moe-nvfp4/verify_logprobs.py compare kernels/moe-nvfp4/results/verify_hybrid.logprobs.json kernels/moe-nvfp4/results/verify_hybrid_b.logprobs.json
```
