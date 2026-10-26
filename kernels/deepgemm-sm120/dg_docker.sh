#!/bin/bash
# usage: dg_docker.sh [--gpu] <cmd...>   runs a command in dg-sm120-env:1 (vllm-openai:v0.32.0 + libdw-dev) (torch 2.13+cu130, nvcc 13.0) with this dir at /work
D=$(cd "$(dirname "$0")" && pwd)
GPUARG=""
if [ "$1" = "--gpu" ]; then GPUARG="--gpus device=0"; shift; fi
NAME=${NAME:-dg-sm120-$$}
exec sg docker -c "docker run --rm --name $NAME $GPUARG -u $(id -u):$(id -g) --entrypoint bash \
  -e HOME=/work/home -e CUDA_HOME=/usr/local/cuda-13.0 -e PATH=/work/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin \
  -e CPLUS_INCLUDE_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include -e LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib -e PYTHONPATH=/work/src -e DG_JIT_CACHE_DIR=/work/home/jit_cache \
  -v $D:/work -v $D/../moe-nvfp4:/moe:ro -v /opt/intramind/hf-cache:/hf:ro -v /opt/intramind/hf-cache:/root/.cache/huggingface:ro -e HF_HUB_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 -w /work/src dg-sm120-env:1 -c '$*'"
