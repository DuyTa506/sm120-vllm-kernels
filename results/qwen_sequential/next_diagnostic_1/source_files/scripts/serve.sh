#!/bin/bash
# Start a vLLM v0.32.0 bench server for unsloth/Qwen3.6-35B-A3B-NVFP4-Fast on one SM120 GPU.
# Model flags mirror the production deployment (MTP-2, prefix caching, async scheduling).
#
# usage: CONFIG=<production|tuned|optimized|next> scripts/serve.sh [extra vllm serve args]
#   production  vLLM defaults for batching and CUDA graph sizes
#   tuned       + --max-num-batched-tokens 16384, CUDA graphs captured up to 16384 tokens
#   optimized   tuned + patches 01/02 with all three env flags on
#   next        optimized + one-kernel FP8 attention (03), 16-token kernel pages under a 2176 block (04),
#               fused GDN conv (05); needs a build/ made from all five patches (scripts/build_patched.sh)
# env: FLAGS     extra docker args, e.g. FLAGS="-e VLLM_SM120_FP8_PREFILL=0" to ablate one patch
#      NSYS=1    run under `nsys launch` so scripts/nsys_capture.sh can record a window
#      NOSPEC=1  drop MTP speculative decoding
#      NAME, PORT, GPU, HF_CACHE, NSYS_DIR
set -e
D=$(cd "$(dirname "$0")/.." && pwd)
CONFIG=${CONFIG:-optimized}
NAME=${NAME:-bench-vllm}
PORT=${PORT:-8100}
GPU=${GPU:-0}
HF_CACHE=${HF_CACHE:-/opt/intramind/hf-cache}
NSYS_DIR=${NSYS_DIR:-/opt/nvidia/nsight-systems/2025.3.2}
IMG=vllm/vllm-openai:v0.32.0
PKG=/usr/local/lib/python3.12/dist-packages/vllm
mkdir -p $D/.cache/vllm $D/.cache/flashinfer $D/results/nsys/raw

DOCKER="-e HF_HUB_OFFLINE=1 -v $HF_CACHE:/root/.cache/huggingface:ro -v $D/.cache/vllm:/root/.cache/vllm"
TUNED="--max-num-batched-tokens 16384 --cudagraph-capture-sizes $(cat $D/scripts/cudagraph_sizes.txt)"
case $CONFIG in
  production) VARGS="" ;;
  tuned) VARGS="$TUNED" ;;
  optimized)
    [ -f $D/build/vllm/v1/attention/backends/flashinfer.py ] || $D/scripts/build_patched.sh
    VARGS="$TUNED"
    DOCKER="$DOCKER -e VLLM_SM120_FP8_PREFILL=1 -e VLLM_FI_KERNEL_PAGE64=1 -e VLLM_GDN_INTERNAL_CKPT=1
      -v $D/.cache/flashinfer:/root/.cache/flashinfer
      -v $D/build/vllm/v1/attention/backends/flashinfer.py:$PKG/v1/attention/backends/flashinfer.py:ro
      -v $D/build/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:$PKG/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:ro"
    ;;
  next)
    [ -f $D/build/vllm/v1/attention/backends/sm120_fp8_fmha_glue.py ] || { echo "build/ predates patches 03-05: run scripts/build_patched.sh" >&2; exit 1; }
    VARGS="$TUNED --block-size 2176"
    B=$D/build/vllm
    DOCKER="$DOCKER -e VLLM_SM120_FP8_PREFILL=1 -e VLLM_SM120_FP8_PREFILL_NOGLUE=1 -e VLLM_FI_KERNEL_PAGES=16
      -e VLLM_GDN_INTERNAL_CKPT=1 -e VLLM_GDN_FUSED_CONV=1
      -v $D/.cache/flashinfer:/root/.cache/flashinfer
      -v $B/v1/attention/backends/flashinfer.py:$PKG/v1/attention/backends/flashinfer.py:ro
      -v $B/v1/attention/backends/sm120_fp8_fmha_glue.py:$PKG/v1/attention/backends/sm120_fp8_fmha_glue.py:ro
      -v $B/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:$PKG/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:ro
      -v $B/model_executor/layers/mamba/ops/fused_conv_post_conv.py:$PKG/model_executor/layers/mamba/ops/fused_conv_post_conv.py:ro"
    ;;
  *) echo "unknown CONFIG=$CONFIG" >&2; exit 1 ;;
esac

SPEC='--speculative-config {"method":"mtp","num_speculative_tokens":2}'
if [ "$NOSPEC" = "1" ]; then SPEC=""; fi
if [ "$NSYS" = "1" ]; then
  ENTRY="--entrypoint /nsys/target-linux-x64/nsys"
  DOCKER="$DOCKER -v $NSYS_DIR:/nsys:ro -v $D/results/nsys/raw:/out"
  PRE="launch -t cuda,nvtx --session-new=vl vllm serve"
else
  ENTRY="--entrypoint vllm"
  PRE="serve"
fi

docker rm -f $NAME >/dev/null 2>&1 || true
docker run -d $DOCKER $FLAGS $ENTRY --name $NAME --gpus device=$GPU --ipc=host -p 127.0.0.1:$PORT:8000 $IMG \
  $PRE unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --served-model-name bench --language-model-only \
  --max-model-len 131072 --max-num-seqs 64 --gpu-memory-utilization 0.78 --enable-chunked-prefill \
  --enable-prefix-caching $SPEC --async-scheduling $VARGS "$@"

echo "waiting for $NAME on :$PORT (CONFIG=$CONFIG)"
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null && { echo ready; exit 0; }
  docker ps -q -f name=^$NAME$ | grep -q . || { docker logs --tail 30 $NAME; exit 1; }
  sleep 5
done
echo "timed out" >&2; exit 1
