#!/usr/bin/env bash
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
TAG=${1:?unique result tag required}; shift
[[ $TAG =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
O="$D/results/$TAG"
[[ ! -e $O ]] || { echo 'result tag already exists' >&2; exit 2; }
LOCK=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
exec 9>"$LOCK"
flock -n 9 || { echo 'GPU lease lock busy' >&2; exit 3; }
mkdir -p "$O/harness" "$O/home"
NAME="pdl-swap-$TAG-$$"
cleanup() {
 local rc=$? clean=0
 trap - EXIT INT TERM
 set +e
 docker rm -f "$NAME" > "$O/cleanup.log" 2>&1
 docker ps -a --filter "name=^/${NAME}$" --format '{{.ID}} {{.Names}}' > "$O/container_after.txt" 2>> "$O/cleanup.log"
 if [[ $? != 0 || -s $O/container_after.txt ]]; then clean=1; fi
 nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$O/gpu_after.txt"
 if [[ $? != 0 || -s $O/gpu_after.txt ]]; then clean=1; fi
 sha256sum --check "$O/sources.sha256" > "$O/source_verify.txt" 2>&1 || clean=1
 if [[ -f $O/source_fingerprint.json ]]; then
  python3 "$O/harness/source_fingerprint.py" "$D/swap_control/src" > "$O/source_fingerprint_after.json" || clean=1
  cmp -s "$O/source_fingerprint.json" "$O/source_fingerprint_after.json" || clean=1
 fi
 printf '%s\n' "$rc" > "$O/exit_code.txt"
 printf '%s\n' "$clean" > "$O/cleanup_status.txt"
 if (( clean != 0 )); then exit 7; fi
 exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$O/gpu_before.txt"
[[ ! -s $O/gpu_before.txt ]] || { echo 'compute GPU occupied' >&2; exit 3; }
FILES=("$D/run_swap_ab_control.sh" "$D/bench_swap_ab.py" "$D/prepare_swap_control.py" "$D/swap_control/manifest.json" "$D/swap_control/native_swap_override.patch" "$D/swap_control/src/csrc/apis/sm120_gemm.hpp" "$D/../deepgemm-sm120/common.py" "$D/../deepgemm-sm120/source_fingerprint.py" "$D/swap_control/build_result.json")
sha256sum "${FILES[@]}" "$D"/swap_control/src/deep_gemm/_C*.so > "$O/sources.sha256"
cp "${FILES[@]}" "$O/harness/"
cp "$D/swap_control/build.log" "$O/harness/build.log"
python3 "$O/harness/source_fingerprint.py" "$D/swap_control/src" > "$O/source_fingerprint.json"
docker image inspect dg-sm120-env:1 --format '{{.Id}}' > "$O/image_id.txt"
CMD=(docker run --rm --name "$NAME" --gpus device=0 -u "$(id -u):$(id -g)" --entrypoint python3
 -e HOME=/out/home -e CUDA_HOME=/usr/local/cuda-13.0
 -e PATH=/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin
 -e CPLUS_INCLUDE_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include
 -e LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib
 -e PYTHONPATH=/work/src -e PYTHONDONTWRITEBYTECODE=1
 -e DG_JIT_CACHE_DIR=/out/jit_cache -e TRITON_CACHE_DIR=/out/triton -e TORCHINDUCTOR_CACHE_DIR=/out/inductor
 -v "$D/swap_control:/work:ro" -v "$D:/harness/kernels/pdl-swapab:ro" -v "$D/../deepgemm-sm120/common.py:/harness/kernels/deepgemm-sm120/common.py:ro"
 -v "$O:/out" -w /harness/kernels/pdl-swapab dg-sm120-env:1
 /harness/kernels/pdl-swapab/bench_swap_ab.py --run-gpu --source-root /work/src --output /out/results.json "$@")
printf '%q ' "${CMD[@]}" > "$O/command.txt"
printf '\n' >> "$O/command.txt"
"${CMD[@]}" 2>&1 | tee "$O/run.log"
