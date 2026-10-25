# Issues found along the way (for vLLM / FlashInfer / NVIDIA)

Environment: RTX PRO 6000 Blackwell Workstation (SM120, 188 SMs, 600 W), `vllm/vllm-openai:v0.32.0`,
FlashInfer 0.7.0.post1, `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast`, MTP-2 speculative decoding.

## FlashInfer b12x MoE backend

1. **Illegal memory access on negative expert ids.** The router can emit `topk_ids = -1` (seen in
   vLLM's dummy/profile run). The b12x grouped kernel indexes with them and faults
   (`CUDA_LAUNCH_BLOCKING=1` + `FLASHINFER_LOGLEVEL=5` localises it to the b12x MoE call).
   Workaround: `patches/vllm-v0.32.0/rejected/b12x-2-mask-negative-topk-ids.patch` maps the
   ids to expert 0 with weight 0. The kernel (or the wrapper) should skip them.
2. **Accuracy.** With the crash worked around, b12x is faster on some shapes but loses quality on this
   NVFP4 checkpoint: prompt NLL +0.03 to +0.10 over the CUTLASS path on long documents, and
   a lower MTP acceptance rate (decode tok/s at c=4: 658 vs 867). See `results/quality/long_eval/`
   and `results/decode/rejected_b12x*.txt`. We did not ship it.

## vLLM

3. **The global `--moe-backend` also applies to the unquantized MTP MoE layer**, which b12x does not
   support. vLLM raises `ValueError` at load. Workaround:
   `rejected/b12x-1-unquantized-mtp-fallback.patch` falls back to `auto` for unquantized layers.
4. **Mamba "align" prefix caching splits every long prompt into two forward passes.** The scheduler
   stops the first chunk at the last block-aligned checkpoint position (8192 tokens → 4256 + 3936),
   so each prefill pays for two passes of the whole model. vLLM already has
   in-forward checkpoints (`num_prefill_checkpoint_blocks`, `prefill_checkpoint_alignment`) and
   FlashInfer's `chunk_gated_delta_rule` can write them (`state_checkpoints`, every 64 tokens), but
   the Qwen GDN layer does not enable them. Patch 02 does so. The kernel output is bit-identical
   with and without checkpoints (checked directly), and a checkpoint-hit vs cold first-token
   distribution test matches the stock path (`eval/ckpt_dist.py`).
5. **Default CUDA graph sizes stop at 512 tokens**, so an 8k prefill runs eagerly and the GPU is
   busy only 40% of the time (`results/nsys/span_8k_production.txt`). Not a bug, but the biggest
   single item: capturing up to 16384 tokens with `--max-num-batched-tokens 16384` cuts the 8k
   prefill GPU span from 369 to 163 ms.
6. **Spec-decode verification steps (query length 1+k) take the prefill path** of the FlashInfer
   backend on SM120, so any prefill-kernel change also changes the verify numerics and MTP acceptance.
   Patch 01 only takes the FP8 kernel when `max_q_len >= 256`.

## FlashInfer SM120 FP8 FMHA (`sm120_fmha_fp8_paged_prefill`)

7. The compiled kernel requires compact `[pages, Hkv, page, D]` K and V tensors. vLLM's cache packs K|V
   in one row (`[blocks, H, page, 2D]`), so a strided view is rejected and patch 01 has to gather
   this batch's pages with `index_select` first. Accepting a row stride would remove the gather.
8. Page sizes must be 16/32/64/128. vLLM's hybrid models use attention block sizes like 2128, which is
   a multiple of the kernel page (16), so passing the kernel page works.

## Tensor-core issue rate (no hidden cap)

`microbench/mma_rate.cu` times `mma.sync` per data type with register-resident operands.
All types run at full rate on this card, short and sustained
(`results/microbench/mma_rate_*.txt`): FP16/BF16 with FP32 accumulate 1022 FLOP/clk/SM
(the same as FP16 accumulate, so there is no GeForce-style half-rate FP32 accumulate), TF32 511,
FP8/INT8 2045, NVFP4 block-scaled 4088.
