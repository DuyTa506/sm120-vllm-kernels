#!/bin/bash
# End-to-end decode A/B: CONFIG=optimized server with/without the SM120 skinny FP8 GEMM patch.
# usage: kernels/dense-fp8-gemm/e2e_decode.sh <on|off> [outfile]
# Takes the GPU lock for one server lifetime (~6-8 min), runs bench/bench_decode.sh, removes the server.
set -e
D=$(cd "$(dirname "$0")/../.." && pwd)
K=$D/kernels/dense-fp8-gemm
MODE=${1:-on}
OUT=${2:-$K/results/e2e_decode_$MODE.txt}
LOCK=${LOCK:-/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock}
PKG=/usr/local/lib/python3.12/dist-packages/vllm
SM=model_executor/kernels/linear/scaled_mm
B=$K/patch/b/vllm/$SM
mkdir -p $K/results
if [ "$MODE" = on ]; then
  F="-e VLLM_SM120_SKINNY_FP8=1 -v $B/cutlass.py:$PKG/$SM/cutlass.py:ro \
     -v $B/sm120_skinny_fp8.py:$PKG/$SM/sm120_skinny_fp8.py:ro -v $B/sm120_skinny_fp8.cu:$PKG/$SM/sm120_skinny_fp8.cu:ro"
else
  F=""
fi
NAME=dense-fp8-gemm-serve
flock $LOCK bash -c "
  trap 'docker rm -f $NAME >/dev/null 2>&1 || true' EXIT
  sg docker -c \"CONFIG=optimized NAME=$NAME PORT=8117 FLAGS='$F' $D/scripts/serve.sh\" &&
  sg docker -c \"$D/bench/bench_decode.sh $NAME\" | tee $OUT &&
  sg docker -c \"docker logs $NAME 2>&1 | grep -i 'skinny\|sm120_fp8' | head -5\" | tee -a $OUT
"
