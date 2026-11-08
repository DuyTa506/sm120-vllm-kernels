#!/usr/bin/env bash
# Run inside the docker group. Owns one disposable server for one experiment.
# Usage: qwen_experiment.sh TAG next|shared|shared_exact|scalar|shared_exact_gdn|scalar_gdn serving|quality|individual|focused|diagnostic|decode_greedy|agentic
# Agentic is a separate final-combination stage; individual never runs it.
set -euo pipefail
D=$(cd "$(dirname "$0")/../../.." && pwd)
HERE=$D/kernels/gdn-replay/round2
TAG=${1:?tag required}
MODE=${2:?shadow preempt off or on required}
case $MODE in shadow|preempt|off|on) ;; *) exit 2;; esac
PREEMPT_GROWTH=${PREEMPT_GROWTH:-0}
[[ $PREEMPT_GROWTH == 0 || $PREEMPT_GROWTH == 1 ]] || exit 2
BOUNDARY_PROMPTS=32; BOUNDARY_CONCURRENCY=16; BOUNDARY_OUTPUT=2304
if [[ $PREEMPT_GROWTH == 1 ]]; then
 [[ $MODE == preempt ]] || { echo "growth profile is preempt-only" >&2; exit 2; }
 BOUNDARY_PROMPTS=8; BOUNDARY_CONCURRENCY=8; BOUNDARY_OUTPUT=6656
fi
[[ -f $HERE/generated/composition.json ]] || { echo "run compose.py CPU preflight first" >&2; exit 2; }
if [[ $MODE == off || $MODE == on ]]; then
  python3 - "$GDN_SHADOW_GATE" "$GDN_PREEMPT_GATE" "$HERE/generated/composition.json" <<'PYGATE'
import json,sys,hashlib
from pathlib import Path
expected=hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest()
for name, mode in zip(sys.argv[1:3], ('shadow','preempt')):
 d=json.load(open(name)); assert d['mode']==mode and d['passed'],name
 assert d['composition_sha256']==expected
 assert (Path(name).parent/'exit_code.txt').read_text().strip()=='0'
PYGATE
fi
SPEC_CONFIG='{"method":"mtp","num_speculative_tokens":2}'
VAR=scalar
SUITE=serving
[[ $TAG =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "invalid tag" >&2; exit 2; }
case $SUITE in serving|quality|individual|focused|diagnostic|decode_greedy|agentic) ;; *) echo "invalid suite" >&2; exit 2;; esac
case $VAR in next|shared|shared_exact|scalar|shared_exact_gdn|scalar_gdn) ;; *) echo "unvalidated/unknown variant" >&2; exit 2;; esac
O=$HERE/results/$TAG
NAME=gdn-r2-$TAG
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
ALLOCATION_MONITOR_PID=""
cleanup() {
  local rc=$?
  if [[ -n $ALLOCATION_MONITOR_PID ]]; then
    kill "$ALLOCATION_MONITOR_PID" 2>/dev/null || true
    wait "$ALLOCATION_MONITOR_PID" 2>/dev/null || true
  fi
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
flags="-e VLLM_CACHE_ROOT=/root/.cache/vllm/gdn-replay-round2-$MODE-$TAG -v $O/serving:/bench-results"
args=(--speculative-config "$SPEC_CONFIG")
source_args=()
if [[ $FI_CONTROL == 1 ]]; then
  # vLLM defaults to immediate engine abort even on docker SIGTERM.
  # Leave time for Worker.shutdown to export the final effective tactic table.
  args+=(--shutdown-timeout 30)
  flags="$flags -v $D/kernels/decode-route-capture:/opt/qwen-fi-control:ro"
  flags="$flags -e PYTHONPATH=/opt/qwen-fi-control:/opt/gdn-round2 -e QWEN_FI_IMPORT_JSON=/opt/qwen-fi-control/references/next_cpu_audit_autotune.json"
  flags="$flags -e QWEN_FI_IMPORT_SHA256=$FI_SOURCE_SHA -e QWEN_FI_IMPORT_OUT=/bench-results/fi_control"
  flags="$flags -e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/bench-results/fi_control/cache/autotune_import"
  source_args+=(--fi-control)
fi
COMPOSITION_SHA=$(sha256sum "$HERE/generated/composition.json" | cut -d' ' -f1)
flags="$flags -v $HERE:/opt/gdn-round2:ro -e GDN_ROUND2_OUT=/bench-results/round2_runtime -e GDN_ROUND2_MODE=$MODE -e GDN_ROUND2_COMPOSITION_SHA=$COMPOSITION_SHA"
G=$PKG/model_executor/layers/mamba/gdn
COMPOSED=$HERE/generated/vllm
flags="$flags -v $COMPOSED/envs.py:$PKG/envs.py:ro -v $COMPOSED/v1/worker/gpu/model_states/mamba_hybrid.py:$PKG/v1/worker/gpu/model_states/mamba_hybrid.py:ro"
flags="$flags -v $COMPOSED/model_executor/layers/mamba/gdn/gdn_state_replay.py:$G/gdn_state_replay.py:ro -v $COMPOSED/model_executor/layers/mamba/gdn/gdn_replay.cu:$G/gdn_replay.cu:ro"
if [[ $MODE == off ]]; then flags="$flags -e VLLM_GDN_STATE_REPLAY=0"; else flags="$flags -e VLLM_GDN_STATE_REPLAY=1 -e VLLM_GDN_STATE_REPLAY_MIN_REQS=32"; fi
if [[ $MODE == shadow || $MODE == preempt ]]; then
 flags="$flags -e VLLM_GDN_STATE_REPLAY_CHECK=2 -e VLLM_GDN_STATE_REPLAY_MIN_REQS=2 -e VLLM_GDN_REPLAY_SHADOW_ROWS=64"
 args+=(-cc.cudagraph_mode=PIECEWISE)
fi
if [[ $MODE == preempt ]]; then
 [[ ${KVBLOCKS:-} =~ ^[1-9][0-9]*$ ]] || { echo "preemption run needs an explicit reviewed KVBLOCKS cap" >&2; exit 2; }
 [[ $KVBLOCKS == 50 ]] || { echo "only reviewed exploratory cap50 prepared" >&2; exit 2; }
 args+=(--num-gpu-blocks-override "$KVBLOCKS" --max-model-len 16384)
 python3 - "$HERE/stress_tokens_cpu.json" "$D/kernels/gdn-replay/tests/stress.py" <<'PYTOKENS'
import hashlib,json,sys
from pathlib import Path
r=json.loads(Path(sys.argv[1]).read_text())
assert r['source_sha256']==hashlib.sha256(Path(sys.argv[2]).read_bytes()).hexdigest()
assert r['cuda_initialized'] is False and r['max_model_len']==16384 and r['max_bound']<=16384
assert len(r['records'])==128 and sum(x['abort'] for x in r['records'])==37
PYTOKENS
fi
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
  bash "$HERE/generated/serve.sh" --gpu-memory-utilization "$GPU_UTIL" \
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

[[ -f $O/serving/round2_runtime/runtime.json ]] || { echo "missing actual runtime guard" >&2; exit 6; }
curl -fsS "$BENCH_URL/metrics" > "$O/metrics_before.txt"
if [[ $MODE == shadow || $MODE == preempt ]]; then
  printf '{"prompts":%s,"concurrency":%s,"input_tokens":128,"output_tokens":%s,"range_ratio":0}\n' "$BOUNDARY_PROMPTS" "$BOUNDARY_CONCURRENCY" "$BOUNDARY_OUTPUT" > "$O/boundary_config.json"
  if [[ $MODE == preempt ]]; then
    python3 "$HERE/observe_allocations.py" --url "$BENCH_URL" --output "$O/allocation_trajectory.jsonl" > "$O/allocation_observer.log" 2>&1 &
    ALLOCATION_MONITOR_PID=$!
  fi
  # Fixed continuations cross one or three2176-token state boundaries.
  docker exec "$NAME" vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --served-model-name bench --dataset-name random --ignore-eos --num-warmups 0 --num-prompts "$BOUNDARY_PROMPTS" --max-concurrency "$BOUNDARY_CONCURRENCY" --random-input-len 128 --random-output-len "$BOUNDARY_OUTPUT" --random-range-ratio 0 --seed 7731 --save-result --save-detailed --result-dir /bench-results --result-filename boundary.json > "$O/boundary_requests.log" 2>&1
  python3 -I "$D/kernels/gdn-replay/tests/stress.py" 128 24 8000 > "$O/stress.log" 2>&1
  python3 "$HERE/check_workload.py" "$O" > "$O/workload_gate.log"
  python3 -I "$D/kernels/gdn-replay/tests/probe.py" "$O/post_stress_probe.json" "$TAG" > "$O/post_stress_probe.log" 2>&1
else
  for concurrency in 48 64; do
    docker exec "$NAME" vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --served-model-name bench --dataset-name random --ignore-eos --num-warmups 4 --num-prompts "$((concurrency*6))" --max-concurrency "$concurrency" --random-input-len 128 --random-output-len 512 --seed "$((900+concurrency))" --save-result --save-detailed --result-dir /bench-results --result-filename "c${concurrency}.json" > "$O/c${concurrency}.log" 2>&1
  done
fi
curl -fsS "$BENCH_URL/metrics" > "$O/metrics_after.txt"
docker logs "$NAME" > "$O/coverage.log" 2>&1
if [[ $MODE == shadow || $MODE == preempt ]]; then
 python3 "$HERE/check_gate.py" "$O" "$MODE"
fi
