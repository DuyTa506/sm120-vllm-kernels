#!/bin/bash
# One GPU job under the shared lock (vLLM image, own container name): cold-weight MoE decode, CUTLASS vs b12x.
# Uses a private copy of the FlashInfer JIT cache (does not touch the shared one).
D=$(cd "$(dirname "$0")/../.." && pwd)
LOCK=${LOCK:-/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock}
FIC=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/fi_cache_pdlswapab
OUT=${OUT:-$D/kernels/pdl-swapab/results/cold_moe_v2.txt}
exec flock "$LOCK" sg docker -c "docker run --rm --name pdlswapab-coldmoe --gpus device=0 --ipc=host -e HF_HUB_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -v /opt/intramind/hf-cache:/root/.cache/huggingface:ro -v $FIC:/root/.cache/flashinfer -v $D:/work -w /work/kernels/moe-nvfp4 \
  --entrypoint timeout vllm/vllm-openai:v0.32.0 600 python3 /work/kernels/pdl-swapab/bench_cold_moe.py $*" > $OUT 2>&1
