#!/bin/bash
# usage: run_in_container.sh [--gpu] <script.py> [args]   (runs inside vllm/vllm-openai:v0.32.0, cwd = this dir)
# --gpu only adds --gpus; the caller must hold the shared GPU lock (flock) for GPU runs.
D=$(cd "$(dirname "$0")" && pwd)
GPU=""
if [ "$1" = "--gpu" ]; then GPU="--gpus device=0"; shift; fi
exec sg docker -c "docker run --rm $GPU --ipc=host --user $(id -u):$(id -g) --entrypoint python3 \
  -e HOME=/tmp -e HF_HUB_OFFLINE=1 \
  -v /opt/intramind/hf-cache:/hf:ro -v $D:/work -w /work vllm/vllm-openai:v0.32.0 $*"
