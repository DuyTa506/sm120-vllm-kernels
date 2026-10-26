#!/bin/bash
# Recreates src/ (DeepGEMM nv_dev b64107f + pinned submodules), builds the C++ extension, and the docker env.
# Does not touch the original DeepGEMM clone (git archive only). Needs network for the two submodules.
set -e
D=$(cd "$(dirname "$0")" && pwd)
DG=${DG:-/home/aiqk3/Desktop/duy_dev/training_space/DeepGEMM}
cd $D
mkdir -p src results home bin && ln -sf /usr/bin/python3 bin/python
git -C $DG archive origin/nv_dev | tar -x -C src
rmdir src/third-party/cutlass src/third-party/deep_jit 2>/dev/null || true
git clone -q https://github.com/NVIDIA/cutlass.git src/third-party/cutlass && git -C src/third-party/cutlass checkout -q f3fde58372d33e9a5650ba7b80fc48b3b49d40c8
git clone -q https://github.com/deepseek-ai/DeepJIT.git src/third-party/deep_jit && git -C src/third-party/deep_jit checkout -q e5bdee2bc4ca519eba00cfc5f0c6e950e6a96a16
# image: vllm/vllm-openai:v0.32.0 (torch 2.13+cu130, nvcc 13.0.88) + libdw-dev (DeepJIT needs elfutils headers)
sg docker -c "docker run --name dg-sm120-prep --entrypoint bash vllm/vllm-openai:v0.32.0 -c 'apt-get update -qq && apt-get install -y -qq libdw-dev' && docker commit dg-sm120-prep dg-sm120-env:1 && docker rm dg-sm120-prep"
./dg_docker.sh 'bash develop.sh'    # ~2 min; produces src/deep_gemm/_C*.so
