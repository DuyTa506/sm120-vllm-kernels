#!/bin/bash
# Run one python script from this directory inside the vLLM image on GPU 0, holding the shared GPU lock.
# usage: ./run_gpu.sh <script.py> [args...]      (container name fused-router-agent, no ports)
D=$(cd "$(dirname "$0")" && pwd)
LOCK=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
BUILD=${FR_BUILD:-/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/fr_build}
HF_CACHE=${HF_CACHE:-/opt/intramind/hf-cache}
NAME=fused-router-agent
SCRIPT=$1; shift
mkdir -p $D/results $BUILD
flock $LOCK sg docker -c "docker run --rm --name $NAME --gpus device=0 --ipc=host --user $(id -u):$(id -g) \
  -e HOME=/tmp -e HF_HUB_OFFLINE=1 -e TORCH_EXTENSIONS_DIR=/build -e FUSED_ROUTER_BUILD=/build/fr \
  -v $HF_CACHE:/hf:ro -v $D:/work -v $D/../decode-roofline:/dr:ro -v $BUILD:/build -w /work \
  --entrypoint timeout vllm/vllm-openai:v0.32.0 \${TIMEOUT:-600} python3 /work/$SCRIPT $*"
