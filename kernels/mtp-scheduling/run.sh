#!/usr/bin/env bash
# Run inside the docker group. Owns one disposable server for one experiment.
# Usage: qwen_experiment.sh TAG next|shared|shared_exact|scalar|shared_exact_gdn|scalar_gdn serving|quality|individual|focused|diagnostic|decode_greedy|agentic
# Agentic is a separate final-combination stage; individual never runs it.
set -euo pipefail
D=$(cd "$(dirname "$0")/../.." && pwd)
HERE=$D/kernels/mtp-scheduling
TAG=${1:?tag required}
PROFILE=${2:?fixed2 or batch21 required}
MTP_DEPTH_PROBE=${MTP_DEPTH_PROBE:-1}
[[ $MTP_DEPTH_PROBE == 0 || $MTP_DEPTH_PROBE == 1 ]] || exit 2
SPEC_CONFIG=$(python3 "$HERE/mtp_schedule_config.py" "$PROFILE")
VAR=scalar
SUITE=serving
[[ $TAG =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "invalid tag" >&2; exit 2; }
case $SUITE in serving|quality|individual|focused|diagnostic|decode_greedy|agentic) ;; *) echo "invalid suite" >&2; exit 2;; esac
case $VAR in next|shared|shared_exact|scalar|shared_exact_gdn|scalar_gdn) ;; *) echo "unvalidated/unknown variant" >&2; exit 2;; esac
O=$HERE/results/$TAG
NAME=qwen-exp-$TAG
PORT=${PORT:-8132}
GPU_UTIL=${GPU_UTIL:-0.78}
GPU=${GPU:-0}
FI_CONTROL=1
[[ $FI_CONTROL == 0 || $FI_CONTROL == 1 ]] || { echo "invalid FI_CONTROL" >&2; exit 2; }
FI_SOURCE_SHA=7b5146f9b3c1e7cc09586c2087988ea8aaf9da1d86da036539a32dec4b335ec3
GREEDY_DECODE=${GREEDY_DECODE:-0}
[[ $GREEDY_DECODE == 0 || $GREEDY_DECODE == 1 ]] || { echo "invalid GREEDY_DECODE" >&2; exit 2; }
if [[ $SUITE == decode_greedy || $SUITE == focused ]]; then GREEDY_DECODE=1; fi
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
  if [[ $FI_CONTROL == 1 ]]; then
    docker stop -t 45 "$NAME" > "$O/stop.log" 2>&1 || true
    if ! python3 "$D/kernels/decode-route-capture/qwen_fi_control_plugin.py" \
      "$O/serving/fi_control" --source-sha256 "$FI_SOURCE_SHA" > "$O/fi_shutdown_verify.json" 2> "$O/fi_shutdown_verify.log"; then
      if (( rc == 0 )); then rc=5; fi
    fi
  fi
  docker logs "$NAME" > "$O/docker.log" 2>&1 || true
  docker rm -f "$NAME" > "$O/remove_container.log" 2>&1 || rc=8
  if docker container inspect "$NAME" > "$O/container_after.json" 2>&1; then
    echo "owned container still exists" > "$O/cleanup_failure.txt"
    rc=8
  fi
  nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv > "$O/gpu_after.csv" 2>&1 || rc=8
  if ! python3 "$HERE/provenance.py" "$HERE" "$O/experiment_sources.json" --verify > "$O/experiment_source_verify.log" 2>&1; then rc=4; fi
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
flags="-e VLLM_CACHE_ROOT=/root/.cache/vllm/mtp-scheduling-$PROFILE-$TAG -v $O/serving:/bench-results"
args=(--speculative-config "$SPEC_CONFIG")
source_args=()
if [[ $FI_CONTROL == 1 ]]; then
  # vLLM defaults to immediate engine abort even on docker SIGTERM.
  # Leave time for Worker.shutdown to export the final effective tactic table.
  args+=(--shutdown-timeout 30)
  flags="$flags -v $D/kernels/decode-route-capture:/opt/qwen-fi-control:ro"
  flags="$flags -e PYTHONPATH=/opt/qwen-fi-control:/opt/mtp-scheduling -e QWEN_FI_IMPORT_JSON=/opt/qwen-fi-control/references/next_cpu_audit_autotune.json"
  flags="$flags -e QWEN_FI_IMPORT_SHA256=$FI_SOURCE_SHA -e QWEN_FI_IMPORT_OUT=/bench-results/fi_control"
  flags="$flags -e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/bench-results/fi_control/cache/autotune_import"
  source_args+=(--fi-control)
fi
flags="$flags -v $HERE:/opt/mtp-scheduling:ro -e MTP_SCHEDULE_TRACE=/bench-results/depth_trace -e MTP_SCHEDULE_PROFILE=$PROFILE -e MTP_DEPTH_PROBE=$MTP_DEPTH_PROBE"
MOE_VAR=${VAR%_gdn}
GDN_OVERLAY=""
if [[ $VAR == *_gdn ]]; then
  GDN_OVERLAY=$D/kernels/gdn-norm-fp8/overlay/vllm/model_executor/layers/mamba/gdn
  flags="$flags -e VLLM_GDN_NORM_FP8_FUSION=1"
  docker run --rm --entrypoint python3 -v "$D/kernels/gdn-norm-fp8:/gdn:ro" \
    vllm/vllm-openai:v0.32.0 /gdn/integration_config.py > "$O/compilation_config.json"
  args+=(--compilation-config "$(cat "$O/compilation_config.json")")
fi
if [[ $MOE_VAR != next ]]; then
  overlay=$K/$MOE_VAR/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py
  flags="$flags -e VLLM_B12X_FIX_SCALES=1 -e VLLM_B12X_SHARED_WORKSPACE=1"
  flags="$flags -v $overlay:$PKG/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:ro"
  flags="$flags -v $K/patched/vllm/model_executor/layers/fused_moe/oracle/unquantized.py:$PKG/model_executor/layers/fused_moe/oracle/unquantized.py:ro"
  args+=(--moe-backend flashinfer_b12x)
fi
if [[ $MOE_VAR == scalar ]]; then flags="$flags -e VLLM_B12X_SCALAR_INPUT_SCALE=1"; fi
printf 'tag=%s\nvariant=%s\nsuite=%s\nconfig=next\ngpu_util=%s\n' "$TAG" "$VAR" "$SUITE" "$GPU_UTIL" > "$O/config.txt"
printf 'greedy_decode=%s\n' "$GREEDY_DECODE" >> "$O/config.txt"
printf 'nll_rounds=%s\n' "$NLL_ROUNDS" >> "$O/config.txt"
printf 'fi_control=%s\n' "$FI_CONTROL" >> "$O/config.txt"
if [[ $FI_CONTROL == 1 ]]; then printf 'fi_source_sha256=%s\n' "$FI_SOURCE_SHA" >> "$O/config.txt"; fi
printf '%s\n' "$SPEC_CONFIG" > "$O/speculative_config.json"
python3 "$HERE/provenance.py" "$HERE" "$O/experiment_sources.json"
git -C "$D" rev-parse HEAD > "$O/commit.txt"
git -C "$D" diff HEAD --binary > "$O/uncommitted.diff"
python3 -I "$D/bench/source_manifest.py" snapshot --root "$D" --output "$O" --variant "$VAR" "${source_args[@]}" > "$O/source_snapshot.log"
docker image inspect vllm/vllm-openai:v0.32.0 --format '{{.Id}}' > "$O/image.txt"
nvidia-smi -i "$GPU" --query-gpu=name,uuid,driver_version,memory.total,memory.used,clocks.sm,power.limit --format=csv > "$O/gpu_before.csv"
NOSPEC=1 CONFIG=next NAME=$NAME PORT=$PORT GPU=$GPU NSYS=0 FLAGS="$flags" GDN_OVERLAY="$GDN_OVERLAY" \
  "$D/scripts/serve.sh" --gpu-memory-utilization "$GPU_UTIL" \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  "${args[@]}" > "$O/start.log" 2>&1
docker logs "$NAME" > "$O/startup.log" 2>&1
if [[ $VAR == *_gdn ]]; then
  python3 -I "$D/bench/verify_gdn_startup.py" "$O/startup.log" > "$O/gdn_readiness.json"
fi
if [[ $FI_CONTROL == 1 ]]; then
  python3 "$D/kernels/decode-route-capture/qwen_fi_control_plugin.py" \
    "$O/serving/fi_control" --source-sha256 "$FI_SOURCE_SHA" --startup-only > "$O/fi_startup_verify.json"
fi
docker inspect "$NAME" > "$O/container.json"
nvidia-smi -i "$GPU" --query-gpu=timestamp,memory.used,utilization.gpu,clocks.sm,power.draw --format=csv --loop=1 > "$O/gpu_during.csv" &
MONITOR_PID=$!
export BENCH_URL=http://127.0.0.1:$PORT
[[ -n $(find "$O/serving/depth_trace" -name 'runtime-*.json' -print -quit) ]] || { echo "missing effective runner/proposer verification" >&2; exit 6; }
python3 "$HERE/mixed.py" --url "$BENCH_URL" --output "$O/mixed" --decode-concurrency "${DECODE_CONCURRENCY:-16}" --seed "${SEED:-73019}"
[[ $MTP_DEPTH_PROBE == 0 || -n $(find "$O/serving/depth_trace" -name 'scheduler-*.jsonl' -print -quit) ]] || { echo "missing actual scheduler depth trace" >&2; exit 7; }
curl -fsS "$BENCH_URL/metrics" > "$O/metrics_after.txt"
python3 "$HERE/provenance.py" "$HERE" "$O/experiment_sources.json" --verify
