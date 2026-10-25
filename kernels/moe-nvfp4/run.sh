#!/bin/bash
# usage: kernels/moe-nvfp4/run.sh <script.py> [args...]   (runs inside the vLLM image, under the GPU lock)
D=$(cd "$(dirname "$0")/../.." && pwd)
LOCK=${GPU_LOCK:-/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock}
S=$1; shift
flock $LOCK sg docker -c "docker run --rm --name moe-nvfp4-$$ --gpus device=0 --ipc=host --entrypoint bash -e HF_HUB_OFFLINE=1 \
  -v /opt/intramind/hf-cache:/root/.cache/huggingface:ro -v $D/.cache/flashinfer:/root/.cache/flashinfer \
  -v $D:/work -w /work/kernels/moe-nvfp4 vllm/vllm-openai:v0.32.0 -c '$( [[ $S == *.sh ]] && echo bash || echo python3 ) $S $*'"
