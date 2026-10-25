#!/bin/bash
# VERIFIER: interleaved in-server A/B of the fused conv path (one server, toggle per request).
# Same server flags as e2e.sh (optimized config + patch 03), but the GDN file is
# verify_qwen_gdn_linear_attn.py whose fused branch is on iff /vflag/fused_on exists.
# usage (repo root, under the GPU lock):  sg docker -c kernels/gdn-conv-fusion/verify_e2e.sh <flagdir>
set -e
FLAGDIR=${1:?flag dir}
D=$(cd "$(dirname "$0")/../.." && pwd)
K=$D/kernels/gdn-conv-fusion
NAME=gdn-conv-fusion-verify-srv
PORT=${PORT:-8117}
IMG=vllm/vllm-openai:v0.32.0
PKG=/usr/local/lib/python3.12/dist-packages/vllm
mkdir -p $FLAGDIR
rm -f $FLAGDIR/fused_on
docker rm -f $NAME >/dev/null 2>&1 || true
trap 'docker rm -f $NAME >/dev/null 2>&1 || true' EXIT
docker run -d --name $NAME --gpus device=0 --ipc=host -p 127.0.0.1:$PORT:8000 \
  -e HF_HUB_OFFLINE=1 -v /opt/intramind/hf-cache:/root/.cache/huggingface:ro \
  -v $D/.cache/vllm:/root/.cache/vllm -v $D/.cache/flashinfer:/root/.cache/flashinfer \
  -e VLLM_SM120_FP8_PREFILL=1 -e VLLM_FI_KERNEL_PAGE64=1 -e VLLM_GDN_INTERNAL_CKPT=1 \
  -e VLLM_GDN_FUSED_CONV=0 \
  -v $D/build/vllm/v1/attention/backends/flashinfer.py:$PKG/v1/attention/backends/flashinfer.py:ro \
  -v $K/verify_qwen_gdn_linear_attn.py:$PKG/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:ro \
  -v $K/build/vllm/model_executor/layers/mamba/ops/fused_conv_post_conv.py:$PKG/model_executor/layers/mamba/ops/fused_conv_post_conv.py:ro \
  -v $FLAGDIR:/vflag:ro \
  --entrypoint vllm $IMG serve unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --served-model-name bench \
  --language-model-only --max-model-len 131072 --max-num-seqs 64 --gpu-memory-utilization 0.78 \
  --enable-chunked-prefill --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --async-scheduling \
  --max-num-batched-tokens 16384 --cudagraph-capture-sizes $(cat $D/scripts/cudagraph_sizes.txt) >/dev/null
for i in $(seq 1 150); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null && break
  docker ps -q -f name=^$NAME$ | grep -q . || { docker logs --tail 40 $NAME; exit 1; }
  sleep 4
done
curl -sf http://127.0.0.1:$PORT/health >/dev/null || { echo "server timeout"; exit 1; }
python3 $K/verify_e2e_client.py http://127.0.0.1:$PORT $FLAGDIR
docker logs $NAME 2>&1 | grep -iE "error|traceback" | head -5 || true
