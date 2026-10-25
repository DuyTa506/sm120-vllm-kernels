#!/bin/bash
# End-to-end check of patch 03 on the "optimized" config (patches 01+02 + their flags).
# usage (from the repo root, under the GPU lock):  kernels/gdn-conv-fusion/e2e.sh <0|1> [tag]
#   arg = VLLM_GDN_FUSED_CONV. Starts a server, records greedy outputs and 8k-prefill
#   TTFT, then removes the container. Results: kernels/gdn-conv-fusion/e2e/fused<arg>.*
# Same model flags as scripts/serve.sh CONFIG=optimized, but the GDN layer file comes
# from kernels/gdn-conv-fusion/build (= build/vllm + patch 03).
set -e
F=${1:-1}
TAG=${2:-}
D=$(cd "$(dirname "$0")/../.." && pwd)
K=$D/kernels/gdn-conv-fusion
NAME=gdn-conv-fusion-srv
PORT=${PORT:-8110}
IMG=vllm/vllm-openai:v0.32.0
PKG=/usr/local/lib/python3.12/dist-packages/vllm
OUT=$K/e2e
mkdir -p $OUT $D/.cache/vllm $D/.cache/flashinfer
docker rm -f $NAME >/dev/null 2>&1 || true
trap 'docker rm -f $NAME >/dev/null 2>&1 || true' EXIT
docker run -d --name $NAME --gpus device=0 --ipc=host -p 127.0.0.1:$PORT:8000 \
  -e HF_HUB_OFFLINE=1 -v /opt/intramind/hf-cache:/root/.cache/huggingface:ro \
  -v $D/.cache/vllm:/root/.cache/vllm -v $D/.cache/flashinfer:/root/.cache/flashinfer \
  -e VLLM_SM120_FP8_PREFILL=1 -e VLLM_FI_KERNEL_PAGE64=1 -e VLLM_GDN_INTERNAL_CKPT=1 \
  -e VLLM_GDN_FUSED_CONV=$F \
  -v $D/build/vllm/v1/attention/backends/flashinfer.py:$PKG/v1/attention/backends/flashinfer.py:ro \
  -v $K/build/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:$PKG/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:ro \
  -v $K/build/vllm/model_executor/layers/mamba/ops/fused_conv_post_conv.py:$PKG/model_executor/layers/mamba/ops/fused_conv_post_conv.py:ro \
  -v $K:/k:ro \
  --entrypoint vllm $IMG serve unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --served-model-name bench \
  --language-model-only --max-model-len 131072 --max-num-seqs 64 --gpu-memory-utilization 0.78 \
  --enable-chunked-prefill --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --async-scheduling \
  --max-num-batched-tokens 16384 --cudagraph-capture-sizes $(cat $D/scripts/cudagraph_sizes.txt) >/dev/null
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null && break
  docker ps -q -f name=^$NAME$ | grep -q . || { docker logs --tail 40 $NAME; exit 1; }
  sleep 5
done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "server timeout"; exit 1; }
docker exec $NAME python3 /k/e2e_client.py > $OUT/fused$F$TAG.outputs.json
B="docker exec $NAME vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --served-model-name bench --dataset-name random --ignore-eos --random-output-len 1 --max-concurrency 1"
$B --random-input-len 8192 --num-warmups 3 --num-prompts 3 --seed 5 >/dev/null 2>&1
# distinct seeds per length: same-seed random prompts share prefixes -> prefix-cache hits
for L in 8192 16384; do
  echo "input $L: $($B --random-input-len $L --num-warmups 0 --num-prompts 20 --seed $((L + 17)) 2>&1 | grep -E 'Mean TTFT|Median TTFT' | tr -s ' ' | tr '\n' ' ')"
done | tee $OUT/fused$F$TAG.ttft.txt
docker logs $NAME 2>&1 | grep -iE "error|traceback" | head -5 || true
