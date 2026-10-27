#!/usr/bin/env bash
# Deferred GPU job. Invoke only when scheduled, as a member of the docker group.
# Usage: bash run.sh TAG [run_capture.py arguments]
set -euo pipefail
TAG=${1:?capture tag required}
shift
[[ $TAG =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'invalid tag' >&2; exit 2; }
D=$(cd "$(dirname "$0")/../.." && pwd)
K=$D/kernels/decode-route-capture
O=$K/results/$TAG
NAME=qwen-route-$TAG-$$
PKG=/usr/local/lib/python3.12/dist-packages/vllm
IMAGE=${IMAGE:-vllm/vllm-openai:v0.32.0}
HF_CACHE=${HF_CACHE:-/opt/intramind/hf-cache}
GPU=${GPU:-0}
LOCK=${GPU_LOCK:-/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock}
case ${GPU_LOCK_HELD:-0} in
  0) exec 9>"$LOCK"; flock 9 ;;
  1)
    # The sequential parent retains its lock for this entire invocation.
    # A separate blocking flock here would deadlock against that parent.
    if (exec 8>"$LOCK"; flock -n -E 75 8); then
      echo 'GPU_LOCK_HELD=1 requires the caller to hold GPU_LOCK' >&2
      exit 2
    else
      status=$?
      [[ $status == 75 ]] || { echo 'cannot verify GPU_LOCK' >&2; exit "$status"; }
    fi
    ;;
  *) echo 'GPU_LOCK_HELD must be 0 or 1' >&2; exit 2 ;;
esac
if [[ -e $O ]]; then
  echo "existing capture: $O; use a fresh tag" >&2
  exit 2
fi
mkdir -p "$O"
cleanup() {
  local rc=$? cid
  trap - EXIT INT TERM
  # Only a container ID returned by this invocation may be removed.
  if [[ -s $O/container.cid ]]; then
    cid=$(cat "$O/container.cid")
    if [[ $cid =~ ^[a-f0-9]{12,64}$ ]]; then
      docker rm -f "$cid" >/dev/null 2>&1 || true
    fi
  fi
  nvidia-smi -i "$GPU" --query-gpu=timestamp,name,uuid,memory.used,utilization.gpu,clocks.sm,power.draw \
    --format=csv > "$O/gpu_after.csv" 2>&1 || true
  nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory \
    --format=csv > "$O/processes_after.csv" 2>&1 || true
  printf '%s\n' "$rc" > "$O/exit_code.txt"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
printf '%q ' "$0" "$TAG" "$@" > "$O/invocation.txt"
printf '\n' >> "$O/invocation.txt"
printf '%s\n' "GPU_LOCK=$LOCK" "GPU_LOCK_HELD=${GPU_LOCK_HELD:-0}" > "$O/lock.txt"
nvidia-smi -i "$GPU" --query-gpu=timestamp,name,uuid,driver_version,memory.total,memory.used,utilization.gpu,clocks.sm,power.draw,power.limit \
  --format=csv > "$O/gpu_before.csv"
processes=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader)
printf '%s\n' "$processes" > "$O/processes_before.txt"
if [[ -n $processes ]]; then
  echo 'GPU has another compute process; no capture started' >&2
  exit 3
fi
patches=(
  v1/attention/backends/flashinfer.py
  v1/attention/backends/sm120_fp8_fmha_glue.py
  model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
  model_executor/layers/mamba/ops/fused_conv_post_conv.py
)
mounts=()
for relative in "${patches[@]}"; do
  [[ -f $D/build/vllm/$relative ]] || { echo "missing CONFIGnext overlay: $relative" >&2; exit 2; }
  mounts+=(-v "$D/build/vllm/$relative:$PKG/$relative:ro")
done
mkdir -p "$O/cache/vllm" "$O/cache/flashinfer"
docker image inspect "$IMAGE" --format '{{.Id}}' > "$O/image.txt"
sha256sum "$K/"*.py "$K/run.sh" > "$O/probe_source_sha256.txt"
for relative in "${patches[@]}"; do
  sha256sum "$D/build/vllm/$relative"
done > "$O/config_next_sha256.txt"
docker run --rm --name "$NAME" --cidfile "$O/container.cid" --gpus "device=$GPU" --ipc=host \
  --entrypoint python3 -e HF_HUB_OFFLINE=1 -e PYTHONDONTWRITEBYTECODE=1 \
  -e VLLM_SM120_FP8_PREFILL=1 -e VLLM_SM120_FP8_PREFILL_NOGLUE=1 \
  -e VLLM_FI_KERNEL_PAGES=16 -e VLLM_GDN_INTERNAL_CKPT=1 -e VLLM_GDN_FUSED_CONV=1 \
  -v "$HF_CACHE:/root/.cache/huggingface:ro" -v "$D:/work:ro" -v "$O:/out" \
  -v "$O/cache/vllm:/root/.cache/vllm" -v "$O/cache/flashinfer:/root/.cache/flashinfer" \
  "${mounts[@]}" -w /work "$IMAGE" \
  /work/kernels/decode-route-capture/run_capture.py --out /out "$@" 2>&1 | tee "$O/run.log"
