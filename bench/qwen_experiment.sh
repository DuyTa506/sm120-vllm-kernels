#!/usr/bin/env bash
# Run inside the docker group. Owns one disposable server for one experiment.
# Usage: qwen_experiment.sh TAG next|shared|shared_exact|scalar serving|quality|individual|diagnostic|decode_greedy|agentic
# Agentic is a separate final-combination stage; individual never runs it.
set -euo pipefail
D=$(cd "$(dirname "$0")/.." && pwd)
TAG=${1:?tag required}
VAR=${2:?variant required}
SUITE=${3:?suite required}
[[ $TAG =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "invalid tag" >&2; exit 2; }
case $SUITE in serving|quality|individual|diagnostic|decode_greedy|agentic) ;; *) echo "invalid suite" >&2; exit 2;; esac
case $VAR in next|shared|shared_exact|scalar) ;; *) echo "unvalidated/unknown variant" >&2; exit 2;; esac
O=$D/results/qwen_sequential/$TAG
NAME=qwen-exp-$TAG
PORT=${PORT:-8132}
GPU_UTIL=${GPU_UTIL:-0.78}
GPU=${GPU:-0}
GREEDY_DECODE=${GREEDY_DECODE:-0}
[[ $GREEDY_DECODE == 0 || $GREEDY_DECODE == 1 ]] || { echo "invalid GREEDY_DECODE" >&2; exit 2; }
if [[ $SUITE == decode_greedy ]]; then GREEDY_DECODE=1; fi
NLL_ROUNDS=${NLL_ROUNDS:-0}
[[ $NLL_ROUNDS =~ ^[0-9]+$ ]] || { echo "invalid NLL_ROUNDS" >&2; exit 2; }
if [[ $SUITE == diagnostic && $GREEDY_DECODE == 0 && $NLL_ROUNDS == 0 ]]; then
  echo "diagnostic requires GREEDY_DECODE=1 or NLL_ROUNDS>0" >&2; exit 2
fi
PKG=/usr/local/lib/python3.12/dist-packages/vllm
K=$D/kernels/moe-nvfp4
LOCK=${GPU_LOCK:-/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock}
[[ ! -e $O ]] || { echo "result directory already exists: $O" >&2; exit 2; }
exec 9>"$LOCK"
flock 9
if docker container inspect "$NAME" >/dev/null 2>&1; then
  echo "container already exists: $NAME" >&2; exit 2
fi
if [[ -n $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader) ]]; then
  echo "GPU has another compute process; no benchmark started" >&2; exit 3
fi
mkdir -p "$O/serving"
MONITOR_PID=""
cleanup() {
  local rc=$?
  if [[ -n $MONITOR_PID ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  docker logs "$NAME" > "$O/docker.log" 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  if [[ -f $O/source_manifest.json ]]; then
    if ! python3 -I "$D/bench/source_manifest.py" verify --root "$D" --output "$O" > "$O/source_verify.log" 2>&1; then
      if (( rc == 0 )); then rc=4; fi
    fi
  fi
  printf '%s\n' "$rc" > "$O/exit_code.txt"
  trap - EXIT
  exit "$rc"
}
trap cleanup EXIT
flags="-e VLLM_CACHE_ROOT=/root/.cache/vllm/qwen-sequential-$VAR -v $O/serving:/bench-results"
args=()
if [[ $VAR != next ]]; then
  overlay=$K/$VAR/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py
  flags="$flags -e VLLM_B12X_FIX_SCALES=1 -e VLLM_B12X_SHARED_WORKSPACE=1"
  flags="$flags -v $overlay:$PKG/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:ro"
  flags="$flags -v $K/patched/vllm/model_executor/layers/fused_moe/oracle/unquantized.py:$PKG/model_executor/layers/fused_moe/oracle/unquantized.py:ro"
  args=(--moe-backend flashinfer_b12x)
fi
if [[ $VAR == scalar ]]; then flags="$flags -e VLLM_B12X_SCALAR_INPUT_SCALE=1"; fi
printf 'tag=%s\nvariant=%s\nsuite=%s\nconfig=next\ngpu_util=%s\n' "$TAG" "$VAR" "$SUITE" "$GPU_UTIL" > "$O/config.txt"
printf 'greedy_decode=%s\n' "$GREEDY_DECODE" >> "$O/config.txt"
printf 'nll_rounds=%s\n' "$NLL_ROUNDS" >> "$O/config.txt"
git -C "$D" rev-parse HEAD > "$O/commit.txt"
git -C "$D" diff HEAD --binary > "$O/uncommitted.diff"
python3 -I "$D/bench/source_manifest.py" snapshot --root "$D" --output "$O" --variant "$VAR" > "$O/source_snapshot.log"
docker image inspect vllm/vllm-openai:v0.32.0 --format '{{.Id}}' > "$O/image.txt"
nvidia-smi -i "$GPU" --query-gpu=name,uuid,driver_version,memory.total,memory.used,clocks.sm,power.limit --format=csv > "$O/gpu_before.csv"
CONFIG=next NAME=$NAME PORT=$PORT GPU=$GPU NSYS=0 FLAGS="$flags" \
  "$D/scripts/serve.sh" --gpu-memory-utilization "$GPU_UTIL" \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  "${args[@]}" > "$O/start.log" 2>&1
docker logs "$NAME" > "$O/startup.log" 2>&1
docker inspect "$NAME" > "$O/container.json"
nvidia-smi -i "$GPU" --query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,power.draw --format=csv --loop=1 > "$O/gpu_during.csv" &
MONITOR_PID=$!
export BENCH_URL=http://127.0.0.1:$PORT
curl -fsS "$BENCH_URL/metrics" > "$O/metrics_before.txt"

if [[ $SUITE == serving || $SUITE == individual ]]; then
  python3 -I "$D/bench/serving_matrix.py" --container "$NAME" \
    --output-dir "$O/serving" --base-url "$BENCH_URL" --suite all \
    > "$O/serving.log" 2>&1
fi
if [[ $SUITE == quality || $SUITE == individual ]]; then
  python3 -I "$K/verify_logprobs.py" dump "$O/logprobs.json" > "$O/logprobs.log" 2>&1
  python3 -I "$K/verify_logprobs.py" dump "$O/logprobs_repeat.json" > "$O/logprobs_repeat.log" 2>&1
  python3 -I "$D/eval/long_eval.py" "$O/long_eval.json" > "$O/long_eval.log" 2>&1
  # Identical tag keeps prompt text equal across variants; each server is new.
  python3 -I "$D/eval/ckpt_dist.py" qwen-sequential > "$O/checkpoint.log" 2>&1
fi
if [[ $SUITE == decode_greedy || $GREEDY_DECODE == 1 ]]; then
  python3 -I "$D/bench/serving_matrix.py" --container "$NAME" \
    --output-dir "$O/serving" --base-url "$BENCH_URL" --suite decode \
    --temperature 0 > "$O/decode_greedy.log" 2>&1
fi
if (( NLL_ROUNDS > 0 )); then
  python3 -I "$D/eval/repeated_nll.py" --output "$O/repeated_nll.json" \
    --rounds "$NLL_ROUNDS" > "$O/repeated_nll.log" 2>&1
fi
if [[ $SUITE == agentic ]]; then
  for set in short long; do
    concs=1,4,8,16,32
    [[ $set == long ]] && concs=1,8,32
    curl -fsS "$BENCH_URL/metrics" > "$O/agentic_${set}_metrics_before.txt"
    python3 -I "$D/bench/agentic_bench.py" replay \
      "$D/results/agentic/trajectories_${set}_production.json.xz" \
      "$O/agentic_${set}.json" "$concs" > "$O/agentic_${set}.log" 2>&1
    curl -fsS "$BENCH_URL/metrics" > "$O/agentic_${set}_metrics_after.txt"
  done
  # Separate live tool-calling smoke from token-exact performance replay.
  # The repository harness simulates writes/tests; it never executes model code.
  python3 -I "$D/bench/agentic_bench.py" record "$O/agentic_live_short.json" short \
    > "$O/agentic_live_short.log" 2>&1
fi
curl -fsS "$BENCH_URL/metrics" > "$O/metrics_after.txt"
nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv > "$O/processes_after.csv"
