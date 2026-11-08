#!/usr/bin/env bash
# Root GPU scheduler only. Owns a process group inside existing dev container.
set -euo pipefail
D=$(cd "$(dirname "$0")/../.." && pwd)
TAG=${1:?unique tag required}
TOKENS=${2:-1024}
MODE=abba
if [[ $TOKENS == geometry ]]; then MODE=geometry; TOKENS=2048; fi
case $TOKENS in 1024|2048|4096) ;; *) echo "unsupported first-screen shape" >&2; exit 2;; esac
[[ $TAG =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
C=${CONTAINER:-qwen-prefill-dev}
GPU=${GPU:-0}
LOCK=${GPU_LOCK:-/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock}
O=$D/kernels/moe-task-order/results/$TAG
CO=/work/kernels/moe-task-order/results/$TAG
[[ ! -e $O ]] || { echo "tag already exists" >&2; exit 2; }
exec 9>"$LOCK"
flock 9
[[ -z $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader) ]] || { echo "GPU occupied" >&2; exit 3; }
mkdir -p "$O"
cleanup() {
  rc=$?
  trap - EXIT INT TERM
  docker exec -e TASK_ORDER_MODE="$MODE" -e TASK_ORDER_TOKENS="$TOKENS" "$C" python3 /work/kernels/moe-task-order/process_guard.py stop "$CO" > "$O/process_cleanup.json" 2>&1 || rc=8
  docker inspect "$C" > "$O/container_after.json" 2>&1 || true
  nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv > "$O/gpu_after.csv" 2>&1 || rc=8
  python3 "$D/kernels/moe-task-order/provenance.py" "$D/kernels/moe-task-order" "$O/launcher_sources.json" --verify > "$O/source_verify.log" 2>&1 || rc=4
  printf '%s\n' "$rc" > "$O/exit_code.txt"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python3 "$D/kernels/moe-task-order/provenance.py" "$D/kernels/moe-task-order" "$O/launcher_sources.json"
docker inspect "$C" > "$O/container_before.json"
git -C "$D" rev-parse HEAD > "$O/commit.txt"
nvidia-smi -i "$GPU" --query-gpu=name,uuid,driver_version,memory.used --format=csv > "$O/gpu_before.csv"
docker exec -e TASK_ORDER_MODE="$MODE" -e TASK_ORDER_TOKENS="$TOKENS" "$C" python3 /work/kernels/moe-task-order/process_guard.py preflight "$CO" > "$O/preflight.json"
docker exec -e TASK_ORDER_MODE="$MODE" -e TASK_ORDER_TOKENS="$TOKENS" "$C" python3 /work/kernels/moe-task-order/process_guard.py run "$CO" > "$O/run.log" 2>&1
