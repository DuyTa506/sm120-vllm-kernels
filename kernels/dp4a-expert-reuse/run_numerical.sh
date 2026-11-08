#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=${1:?unique tag};M=${2:-1}
[[ $TAG =~ ^[a-zA-Z0-9_-]+$ && $M =~ ^(1|2|3|4|8)$ ]] || exit 2
O=$HERE/results/$TAG
[[ ! -e $O ]] || exit 2
LOCK=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
exec 9>"$LOCK";flock 9
compute=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[[ -z $compute ]] || { echo GPU_busy;exit 3; }
mkdir -p "$O"
python3 "$HERE/provenance.py" "$HERE" "$O/sources.json"
C=/work/kernels/dp4a-expert-reuse/results/$TAG
cleanup() {
 rc=$?;set +e
 docker exec qwen-prefill-dev python3 /work/kernels/dp4a-expert-reuse/cleanup_child.py "$C" > "$O/child_after.json"
 child_rc=$?;((child_rc==0)) || rc=8
 python3 "$HERE/provenance.py" "$HERE" "$O/sources.json" --verify > "$O/source_verify.log" 2>&1 || rc=4
 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$O/gpu_after.csv" || rc=8
 echo "$rc" > "$O/exit_code.txt"
 trap - EXIT;exit "$rc"
}
trap cleanup EXIT
docker inspect qwen-prefill-dev > "$O/container.json"
docker exec qwen-prefill-dev python3 -c 'from pathlib import Path; p=Path("/work/kernels/moe-nvfp4/data/moe_inputs.pt"); assert p.stat().st_size>1000000' > "$O/capture_preflight.log" 2>&1
docker exec -e CUTE_DSL_CACHE_DIR="$C/cache/cute" -e TORCHINDUCTOR_CACHE_DIR="$C/cache/inductor" -e TRITON_CACHE_DIR="$C/cache/triton" qwen-prefill-dev python3 /work/kernels/dp4a-expert-reuse/numerical.py --capture /work/kernels/moe-nvfp4/data/moe_inputs.pt --output "$C/screen.json" --tokens "$M" > "$O/process.log" 2>&1
