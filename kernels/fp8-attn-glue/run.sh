#!/bin/bash
# usage: kernels/fp8-attn-glue/run.sh <script.py> [args]   (runs inside vllm/vllm-openai:v0.32.0 under the GPU lock)
LOCK=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
R=/home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels
S=$1; shift
exec flock $LOCK sg docker -c "docker run --rm --name fp8-attn-glue-$$ --gpus device=0 --ipc=host --entrypoint python3 -e HF_HUB_OFFLINE=1 \
  -v /opt/intramind/hf-cache:/root/.cache/huggingface:ro -v $R:/work \
  -v $R/kernels/fp8-attn-glue/.cache/flashinfer:/root/.cache/flashinfer \
  -w /work/kernels/fp8-attn-glue vllm/vllm-openai:v0.32.0 $S $*"
