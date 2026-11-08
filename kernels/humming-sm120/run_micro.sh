#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
D=$(cd "$HERE/../.." && pwd)
TAG=${1:?unique tag};LAYOUT=${2:-indexed};TOKENS=${3:-1024};JOB=${4:-screen};LAYER=${5:-19}
[[ $LAYER =~ ^(0|19|39)$ ]] || exit 2
case $JOB in screen|screen_rn|numerical_ru|numerical_rn) ;; *) exit 2;; esac
PROGRAM=micro.py;EXTRA=()
if [[ $JOB == numerical_* ]]; then PROGRAM=numerical.py;EXTRA=(--rounding "${JOB#numerical_}");fi
if [[ $JOB == screen_rn ]]; then EXTRA=(--rounding rn);fi
[[ $TAG =~ ^[a-zA-Z0-9_-]+$ && $LAYOUT =~ ^(indexed|grouped)$ && $TOKENS =~ ^(32|1024|4096)$ ]] || exit 2
O=$HERE/results/$TAG
[[ ! -e $O ]] || exit 2
LOCK=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
exec 9>"$LOCK"; flock 9
compute=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z $compute ]] || { echo GPU_busy; exit 3; }
mkdir -p "$O"
python3 "$HERE/provenance.py" "$HERE" "$O/sources.json"
C=/work/kernels/humming-sm120/results/$TAG
cleanup() {
 rc=$?
 set +e
 # Synchronous docker exec is awaited. Record exact tagged child absence; no
 # container deletion, no broad kill patterns, no installed source mutations.
 docker exec qwen-prefill-dev python3 /work/kernels/humming-sm120/cleanup_child.py "$C" > "$O/child_after.json"
 child_rc=$?; ((child_rc==0)) || rc=8
 python3 "$HERE/provenance.py" "$HERE" "$O/sources.json" --verify > "$O/source_verify.log" 2>&1 || rc=4
 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$O/gpu_after.csv" || rc=8
 echo "$rc" > "$O/exit_code.txt"
 trap - EXIT; exit "$rc"
}
trap cleanup EXIT
docker inspect qwen-prefill-dev > "$O/container.json"
docker exec -e HUMMING_CACHE_DIR="$C/cache/humming" -e CUTE_DSL_CACHE_DIR="$C/cache/cute" -e TORCHINDUCTOR_CACHE_DIR="$C/cache/inductor" -e TRITON_CACHE_DIR="$C/cache/triton" qwen-prefill-dev python3 "/work/kernels/humming-sm120/$PROGRAM" --capture /work/kernels/moe-nvfp4/data/moe_inputs.pt --output "$C/screen.json" --layout "$LAYOUT" --tokens "$TOKENS" --layer "$LAYER" "${EXTRA[@]}" > "$O/process.log" 2>&1
