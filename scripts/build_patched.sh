#!/bin/bash
# Extract the pristine vLLM files from the image and apply patches 01-05 into build/vllm/ (all env-gated, default off).
# The patched files are then bind-mounted over the image's copies by serve.sh (CONFIG=optimized).
set -e
D=$(cd "$(dirname "$0")/.." && pwd)
IMG=${IMG:-vllm/vllm-openai:v0.32.0}
PKG=/usr/local/lib/python3.12/dist-packages/vllm
B=$D/build
rm -rf $B && mkdir -p $B
for rel in v1/attention/backends/flashinfer.py model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py; do
  mkdir -p $B/vllm/$(dirname $rel)
  docker run --rm --entrypoint cat $IMG $PKG/$rel > $B/vllm/$rel
done
for p in $D/patches/vllm-v0.32.0/*.patch; do
  echo "applying $(basename $p)"
  patch -s -p1 -d $B < $p
done
