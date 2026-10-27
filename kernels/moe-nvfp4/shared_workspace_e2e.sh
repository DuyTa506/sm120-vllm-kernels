#!/usr/bin/env bash
# Run as a user in the docker group (e.g. sg docker -c '...').
# Usage: shared_workspace_e2e.sh <tag> <cutlass|shared>
# Each variant uses its own compile cache. Existing servers are never touched.
set -euo pipefail
TAG=${1:?tag required}
VAR=${2:?cutlass or shared required}
[[ $TAG =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "invalid tag" >&2; exit 2; }
D=$(cd "$(dirname "$0")/../.." && pwd)
K=$D/kernels/moe-nvfp4
O=$K/results/shared_$TAG
PKG=/usr/local/lib/python3.12/dist-packages/vllm
NAME=qwen-prefill-ab-$TAG
PORT=${PORT:-8132}
CONFIG=${CONFIG:-next}
NSYS=${NSYS:-0}
GPU_UTIL=${GPU_UTIL:-0.60}
LOCK=${GPU_LOCK:-/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock}
mkdir -p "$O"
exec 9>"$LOCK"
flock 9
flags="-e VLLM_CACHE_ROOT=/root/.cache/vllm/qwen-prefill-$VAR -v $O:/bench-results"
args=()
case $VAR in
  cutlass) ;;
  shared)
    flags="$flags -e VLLM_B12X_FIX_SCALES=1 -e VLLM_B12X_SHARED_WORKSPACE=1"
    flags="$flags -v $K/shared/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:$PKG/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:ro"
    flags="$flags -v $K/patched/vllm/model_executor/layers/fused_moe/oracle/unquantized.py:$PKG/model_executor/layers/fused_moe/oracle/unquantized.py:ro"
    args=(--moe-backend flashinfer_b12x)
    ;;
  *) echo "unknown variant $VAR" >&2; exit 2 ;;
esac
cleanup() {
  docker logs "$NAME" > "$O/docker.log" 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT
printf 'variant=%s config=%s gpu_util=%s nsys=%s\n' "$VAR" "$CONFIG" "$GPU_UTIL" "$NSYS" > "$O/summary.txt"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu,clocks.sm,power.draw --format=csv > "$O/gpu_before.csv"
CONFIG=$CONFIG NAME=$NAME PORT=$PORT NSYS=$NSYS FLAGS="$flags" \
  "$D/scripts/serve.sh" --gpu-memory-utilization "$GPU_UTIL" "${args[@]}" > "$O/start.log" 2>&1
docker logs "$NAME" > "$O/startup.log" 2>&1
rg 'KV cache|Available KV|Model loading took|Graph capturing finished|NvFp4 MoE backend' "$O/startup.log" >> "$O/summary.txt" || true
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$O/metrics_before.txt"

bench() {
  local name=$1
  shift
  docker exec "$NAME" vllm bench serve --backend vllm \
    --base-url http://127.0.0.1:8000 --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast \
    --served-model-name bench --dataset-name random --ignore-eos \
    --save-result --result-dir /bench-results --result-filename "$name.json" \
    "$@" > "$O/$name.log" 2>&1
  printf '\n%s\n' "$name" >> "$O/summary.txt"
  rg 'Successful requests|Mean TTFT|Median TTFT|P99 TTFT|Output token throughput|Median ITL' "$O/$name.log" >> "$O/summary.txt"
}
for length in 8192 16384; do
  for seed in 101 202; do
    bench "prefill_${length}_${seed}" --num-warmups 2 --num-prompts 16 \
      --seed "$seed" --max-concurrency 1 --random-input-len "$length" --random-output-len 1
  done
done
bench decode_c1 --num-warmups 2 --num-prompts 8 --seed 501 --max-concurrency 1 \
  --random-input-len 128 --random-output-len 256
bench decode_c4 --num-warmups 2 --num-prompts 16 --seed 504 --max-concurrency 4 \
  --random-input-len 128 --random-output-len 256
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$O/metrics_after.txt"
nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm,power.draw --format=csv > "$O/gpu_after.csv"
if [[ $NSYS == 1 ]]; then
  "$D/scripts/nsys_capture.sh" "shared_${TAG}_8k" 8192 "$NAME" > "$O/nsys8k.txt" 2>&1
fi
cat "$O/summary.txt"
