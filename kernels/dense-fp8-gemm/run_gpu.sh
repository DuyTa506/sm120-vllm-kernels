#!/bin/bash
# usage: kernels/dense-fp8-gemm/run_gpu.sh <script.py> [args...]   (runs inside vllm/vllm-openai:v0.32.0
# under the shared GPU lock; the python process is time-limited inside the container)
# env: LOCK (GPU lock file), EXTRA (extra docker args, e.g. bind mounts), TMO (seconds, default 900)
D=$(cd "$(dirname "$0")/../.." && pwd)
LOCK=${LOCK:-/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock}
TMO=${TMO:-900}
exec flock "$LOCK" sg docker -c "docker run --rm --name dense-fp8-gemm-bench-\$\$ --gpus device=0 --ipc=host \
  -u $(id -u):$(id -g) -e HOME=/tmp -e HF_HUB_OFFLINE=1 $EXTRA \
  -v /opt/intramind/hf-cache:/root/.cache/huggingface:ro -v /opt/intramind/hf-cache:/hf:ro -v $D:/work -w /work \
  --entrypoint timeout vllm/vllm-openai:v0.32.0 $TMO python3 $*"
