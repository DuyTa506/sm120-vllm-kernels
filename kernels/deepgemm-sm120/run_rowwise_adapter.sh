#!/usr/bin/env bash
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
TAG=${1:?unique result tag required}; shift
[[ "$TAG" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
SRC=${DG_SOURCE_DIR:-/home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/deepgemm-sm120/src}
OUT="$D/results/$TAG"
ENTRY=${DG_ENTRYPOINT:-bench_rowwise_adapter.py}
[[ "$ENTRY" == bench_rowwise_adapter.py || "$ENTRY" == capture_gdn_operands.py || "$ENTRY" == replay_gdn_operands.py ]] || exit 2
LOCK=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
exec 9>"$LOCK"
flock -n 9 || { echo 'GPU benchmark lock busy'; exit 1; }
mkdir "$OUT"
mkdir "$OUT/home" "$OUT/harness"
NAME="dg-rowwise-$TAG-$$"
cleanup() {
    local run_status=$? cleanup_status=0
    trap - EXIT INT TERM
    set +e
    printf '%s\n' "$run_status" > "$OUT/exit_code.txt"
    # Only remove this launcher's unique container, then query actual daemon state.
    sg docker -c "docker rm -f '$NAME'" > "$OUT/cleanup.log" 2>&1
    sg docker -c "docker ps -a --filter 'name=^/${NAME}$' --format '{{.ID}} {{.Names}}'" > "$OUT/container_after.txt" 2>> "$OUT/cleanup.log"
    if [[ $? -ne 0 || -s "$OUT/container_after.txt" ]]; then cleanup_status=1; fi
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$OUT/gpu_after.txt" 2>&1
    if [[ $? -ne 0 ]]; then cleanup_status=1; fi
    sha256sum --check "$OUT/harness.sha256" > "$OUT/harness_verify_after.txt" 2>&1
    if [[ $? -ne 0 ]]; then cleanup_status=1; fi
    if [[ -f "$OUT/source_fingerprint.json" ]]; then
        python3 "$OUT/harness/source_fingerprint.py" "$SRC" > "$OUT/source_fingerprint_after.json"
        if [[ $? -ne 0 ]] || ! cmp -s "$OUT/source_fingerprint.json" "$OUT/source_fingerprint_after.json"; then cleanup_status=1; fi
    fi
    printf '%s\n' "$cleanup_status" > "$OUT/cleanup_status.txt"
    if [[ "$cleanup_status" -ne 0 ]]; then
        echo "Cleanup/evidence verification failed; inspect $OUT" >&2
        exit 1
    fi
    exit "$run_status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$OUT/gpu_before.txt"
if [[ -s "$OUT/gpu_before.txt" ]]; then
    echo 'GPU has compute processes; refusing launch' >&2
    exit 1
fi
FILES=("$D/bench_rowwise_adapter.py" "$D/common.py" "$D/run_rowwise_adapter.sh" "$D/source_fingerprint.py" "$D/capture_gdn_operands.py" "$D/replay_gdn_operands.py")
cp "${FILES[@]}" "$OUT/harness/"
cp "$D/../moe-nvfp4/data/prompt.txt" "$OUT/harness/prompt.txt"
sha256sum "${FILES[@]}" "$D/../moe-nvfp4/data/prompt.txt" > "$OUT/harness.sha256"
git -C "$SRC" rev-parse HEAD > "$OUT/enclosing_repo_revision.txt"
python3 "$OUT/harness/source_fingerprint.py" "$SRC" > "$OUT/source_fingerprint.json"
git -C "$SRC" diff > "$OUT/enclosing_repo_diff.patch"
sg docker -c 'docker image inspect dg-sm120-env:1 --format "{{.Id}}"' > "$OUT/image_id.txt"
EXTRA_MOUNTS=(-v /opt/intramind/hf-cache:/hf:ro)
if [[ -n "${DG_INPUT_DIR:-}" ]]; then EXTRA_MOUNTS+=(-v "$DG_INPUT_DIR:/inputs:ro"); fi
if [[ -n "${DG_SKINNY_DIR:-}" ]]; then EXTRA_MOUNTS+=(-v "$DG_SKINNY_DIR:/skinny:ro"); fi
CMD=(docker run --rm --name "$NAME" --gpus device=0 -u "$(id -u):$(id -g)" --entrypoint python3
 -e HOME=/out/home -e CUDA_HOME=/usr/local/cuda-13.0
 -e PATH=/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin
 -e CPLUS_INCLUDE_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include
 -e LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib
 -e PYTHONPATH=/work/src:/harness -e PYTHONDONTWRITEBYTECODE=1
 -e DG_JIT_CACHE_DIR=/out/jit_cache -e TRITON_CACHE_DIR=/out/triton -e TORCHINDUCTOR_CACHE_DIR=/out/inductor
 "${EXTRA_MOUNTS[@]}" -e HF_HUB_OFFLINE=1
 -v "$SRC:/work/src:ro" -v "$OUT/harness:/harness:ro" -v "$OUT:/out" -w /harness dg-sm120-env:1
 /harness/"$ENTRY" --run-gpu --out /out/results.json "$@")
printf -v QUOTED '%q ' "${CMD[@]}"
sg docker -c "$QUOTED" 2>&1 | tee "$OUT/run.log"
