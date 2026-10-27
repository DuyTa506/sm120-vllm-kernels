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
MONITOR_PID=""
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
  if [[ -n $MONITOR_PID ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  docker logs "$NAME" > "$O/docker.log" 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT
printf 'protocol=cold-v2 variant=%s config=%s gpu_util=%s nsys=%s\n' "$VAR" "$CONFIG" "$GPU_UTIL" "$NSYS" > "$O/summary.txt"
nvidia-smi --query-gpu=name,memory.used,utilization.gpu,clocks.sm,power.draw --format=csv > "$O/gpu_before.csv"
CONFIG=$CONFIG NAME=$NAME PORT=$PORT NSYS=$NSYS FLAGS="$flags" \
  "$D/scripts/serve.sh" --gpu-memory-utilization "$GPU_UTIL" "${args[@]}" > "$O/start.log" 2>&1
docker logs "$NAME" > "$O/startup.log" 2>&1
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,power.draw --format=csv --loop=1 > "$O/gpu_during.csv" &
MONITOR_PID=$!
rg 'KV cache|Available KV|Model loading took|Graph capturing finished|NvFp4 MoE backend' "$O/startup.log" >> "$O/summary.txt" || true
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$O/metrics_before.txt"

bench() {
  local name=$1
  shift
  docker exec "$NAME" vllm bench serve --backend vllm \
    --base-url http://127.0.0.1:8000 --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast \
    --served-model-name bench --dataset-name random --ignore-eos \
    --ready-check-timeout-sec 0 --save-result --save-detailed \
    --result-dir /bench-results --result-filename "$name.json" \
    "$@" > "$O/$name.log" 2>&1
  printf '\n%s\n' "$name" >> "$O/summary.txt"
  rg 'Successful requests|Mean TTFT|Median TTFT|P99 TTFT|Output token throughput|Median ITL' "$O/$name.log" >> "$O/summary.txt"
}
for length in 8192 16384; do
  for seed in 101 202; do
    # The CLI's readiness/warmup request uses input_requests[0]. Warm with
    # disjoint seeds and disable both on the measured invocation so its
    # first request cannot reuse the warmup prefix. Include the length in
    # the seed: equal seeds at different lengths share the same prefix.
    bench "warmup_${length}_${seed}" --num-warmups 0 --num-prompts 2 \
      --seed "$((length + seed + 100000))" --max-concurrency 1 --random-input-len "$length" --random-output-len 1
    bench "prefill_${length}_${seed}" --num-warmups 0 --num-prompts 16 \
      --seed "$((length + seed))" --max-concurrency 1 --random-input-len "$length" --random-output-len 1
  done
done
bench decode_warmup_c1 --num-warmups 0 --num-prompts 2 --seed 1501 --max-concurrency 1 \
  --random-input-len 128 --random-output-len 64
bench decode_c1 --num-warmups 0 --num-prompts 8 --seed 501 --max-concurrency 1 \
  --random-input-len 128 --random-output-len 256
bench decode_warmup_c4 --num-warmups 0 --num-prompts 8 --seed 1504 --max-concurrency 4 \
  --random-input-len 128 --random-output-len 64
bench decode_c4 --num-warmups 0 --num-prompts 16 --seed 504 --max-concurrency 4 \
  --random-input-len 128 --random-output-len 256
curl -fsS "http://127.0.0.1:$PORT/metrics" > "$O/metrics_after.txt"
nvidia-smi --query-gpu=memory.used,utilization.gpu,clocks.sm,power.draw --format=csv > "$O/gpu_after.csv"
if [[ $NSYS == 1 ]]; then
  # Disjoint warmup and no readiness request inside the capture window.
  bench profile_warmup --num-warmups 0 --num-prompts 2 --seed 1777 \
    --max-concurrency 1 --random-input-len 8192 --random-output-len 1
  trace=shared_${TAG}_8k
  profiler=/nsys/target-linux-x64/nsys
  docker exec "$NAME" "$profiler" start --session=vl -o "/out/$trace" --force-overwrite=true > "$O/nsys_start.log" 2>&1
  bench profile_8k --num-warmups 0 --num-prompts 4 --seed 777 \
    --max-concurrency 1 --random-input-len 8192 --random-output-len 1
  docker exec "$NAME" "$profiler" stop --session=vl > "$O/nsys_stop.log" 2>&1
  nsys export --type sqlite --force-overwrite=true \
    -o "$D/results/nsys/raw/$trace.sqlite" "$D/results/nsys/raw/$trace.nsys-rep" > "$O/nsys_export.log" 2>&1
  python3 -I "$D/bench/nsys_analyze.py" "$D/results/nsys/raw/$trace.sqlite" > "$O/nsys8k.txt" 2>&1
fi
cat "$O/summary.txt"
