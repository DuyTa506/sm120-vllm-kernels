#!/usr/bin/env bash
# Exactly one explicitly root-granted numerical run, no timing or autochain.
set -euo pipefail
D=$(cd "$(dirname "$0")/../.." && pwd)
HERE=$D/kernels/fi-static-scale-audit
MODE=${2:-probe}; [[ $MODE =~ ^(probe|buckets)$ ]] || exit 2
TOKENS=${3:-4}; [[ $TOKENS =~ ^(4|8)$ ]] || exit 2
TAG=${1:?unique tag}; [[ $TAG =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
C=${CONTAINER:-qwen-prefill-dev}; GPU=${GPU:-0}
O=$HERE/results/$TAG; CO=/work/kernels/fi-static-scale-audit/results/$TAG
LOCK=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
[[ ! -e $O ]] || exit 2
exec 9>"$LOCK";flock -n 9
[[ -z $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader) ]] || exit 3
mkdir -p "$O"
cleanup() {
 rc=$?;trap - EXIT INT TERM;set +e
 docker exec "$C" python3 /work/kernels/fi-static-scale-audit/process_guard.py stop "$CO" > "$O/process_cleanup.json" 2>&1 || rc=8
 docker exec -e CUDA_VISIBLE_DEVICES= "$C" python3 /work/kernels/fi-static-scale-audit/process_guard.py verify "$CO" > "$O/runtime_verify.json" 2>&1 || rc=4
 python3 "$HERE/provenance.py" "$HERE" "$O/source_manifest.json" --verify > "$O/source_verify.log" 2>&1 || rc=4
 docker inspect "$C" > "$O/container_after.json" 2>&1 || rc=8
 nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$O/gpu_after.csv" 2>&1 || rc=8
 [[ ! -s $O/gpu_after.csv ]] || rc=8
 printf '%s\n' "$rc" > "$O/exit_code.txt";exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python3 "$HERE/provenance.py" "$HERE" "$O/source_manifest.json"
docker inspect "$C" > "$O/container_before.json"
docker inspect "$C" --format '{{.Image}}' > "$O/image.txt"
git -C "$D" rev-parse HEAD > "$O/commit.txt"
nvidia-smi -i "$GPU" --query-gpu=name,uuid,driver_version,memory.used --format=csv > "$O/gpu_before.csv"
docker exec -e CUDA_VISIBLE_DEVICES= "$C" python3 /work/kernels/fi-static-scale-audit/process_guard.py preflight "$CO" > "$O/preflight.json" 2>&1
docker exec -e AUDIT_EXPERIMENT="$MODE" -e AUDIT_TOKENS="$TOKENS" "$C" python3 /work/kernels/fi-static-scale-audit/process_guard.py run "$CO" > "$O/run.log" 2>&1
