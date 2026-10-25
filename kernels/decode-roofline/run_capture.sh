#!/bin/bash
# One GPU-lock hold: start the optimized server under nsys, run capture.sh, remove the container.
# usage: kernels/decode-roofline/run_capture.sh <tag> [node|graph] [skinny]
#   node   = --cuda-graph-trace=node (per-kernel durations inside CUDA graphs; inflates gaps)
#   graph  = default graph-level trace (accurate replay spans and host gaps)
#   skinny = also mount the patched layers/utils.py + skinny_bf16_gemm.py with VLLM_SKINNY_BF16_GEMM=1
D=$(cd "$(dirname "$0")" && pwd)
TAG=$1; MODE=${2:-graph}; VARIANT=${3:-base}
LOCK=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
PKG=/usr/local/lib/python3.12/dist-packages/vllm
OPTS=""; [ "$MODE" = node ] && OPTS="--cuda-graph-trace=node"
EXTRA=""
if [ "$VARIANT" = skinny ]; then
  EXTRA="-e VLLM_SKINNY_BF16_GEMM=1 -v $D/vllm/model_executor/layers/utils.py:$PKG/model_executor/layers/utils.py:ro -v $D/vllm/model_executor/layers/skinny_bf16_gemm.py:$PKG/model_executor/layers/skinny_bf16_gemm.py:ro"
fi
exec flock $LOCK sg docker -c "
  FLAGS='$EXTRA' CONFIG=\${CONFIG:-optimized} NSYS=1 NAME=decode-agent PORT=8105 NSYS_OPTS='$OPTS' $D/serve_nsys.sh && $D/capture.sh $TAG decode-agent 8105;
  docker logs decode-agent 2>&1 | grep -iE 'SKINNY|Traceback|Error' | head -5;
  docker rm -f decode-agent >/dev/null"
