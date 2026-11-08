#!/usr/bin/env bash
# Exactly one granted control per invocation; ABBA is scheduled by root, never autochained.
set -euo pipefail
D=$(cd "$(dirname "$0")/../.." && pwd)
HERE=$D/kernels/moe-task-order-serving
TAG=${1:?unique tag}; PROFILE=${2:?off or on}
[[ $TAG =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
case $PROFILE in off) LONG_FIRST=0;; on) LONG_FIRST=1;; *) exit 2;; esac
CACHE_FAMILY=${CACHE_FAMILY:?shared paired cache family required}
[[ $CACHE_FAMILY =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
O=$HERE/results/$TAG;NAME=task-order-serving-$TAG;PORT=${PORT:-8135};GPU=${GPU:-0}
PKG=/usr/local/lib/python3.12/dist-packages/vllm
FI_SOURCE_SHA=7b5146f9b3c1e7cc09586c2087988ea8aaf9da1d86da036539a32dec4b335ec3
LOCK=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
[[ ! -e $O ]] || exit 2
exec 9>"$LOCK";flock -n 9 || exit 3
[[ -z $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader) ]] || exit 3
if docker container inspect "$NAME" >/dev/null 2>&1;then exit 3;fi
mkdir -p "$O/serving" "$O/traces"
cleanup() {
 local rc=$?
 trap - EXIT INT TERM
 set +e
 local natural=0
 if [[ -f $O/serving/api_process.json ]];then
  docker exec "$NAME" python3 /opt/task-order/graceful_shutdown.py signal /bench-results/api_process.json > "$O/api_signal.json" 2> "$O/api_signal.log"
  if [[ $? == 0 ]];then
   for attempt in $(seq 1 60);do
    state=$(docker inspect --format '{{.State.Running}}' "$NAME" 2>/dev/null)
    if [[ $state == false ]];then natural=1;break;fi
    sleep 1
   done
  fi
 fi
 printf '%s\n' "$natural" > "$O/natural_container_exit.txt"
 if [[ $natural != 1 ]];then
  printf '%s\n' 'verified API graceful exit unavailable or timed out; owned-container fallback' > "$O/shutdown_fallback.txt"
  docker stop -t 45 "$NAME" > "$O/stop.log" 2>&1
 fi
 python3 "$D/kernels/decode-route-capture/qwen_fi_control_plugin.py" "$O/serving/fi_control" --source-sha256 "$FI_SOURCE_SHA" > "$O/fi_shutdown_verify.json" 2> "$O/fi_shutdown_verify.log"
 if [[ $? -ne 0 && $rc == 0 ]];then rc=5;fi
 docker logs "$NAME" > "$O/docker.log" 2>&1
 docker rm -f "$NAME" > "$O/remove_container.log" 2>&1
 docker ps -a --filter "name=^/${NAME}$" --format '{{.ID}} {{.Names}}' > "$O/container_after.txt"
 if [[ $? -ne 0 || -s $O/container_after.txt ]];then rc=8;fi
 nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$O/gpu_after.csv"
 if [[ $? -ne 0 || -s $O/gpu_after.csv ]];then rc=8;fi
 python3 "$HERE/provenance.py" "$HERE" "$O/experiment_sources.json" --verify > "$O/experiment_source_verify.log" 2>&1 || rc=4
 python3 -I "$D/bench/source_manifest.py" verify --root "$D" --output "$O" > "$O/source_verify.log" 2>&1 || rc=4
 printf '%s\n' "$rc" > "$O/exit_code.txt"
 exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python3 "$HERE/provenance.py" "$HERE" "$O/experiment_sources.json"
python3 -I "$D/bench/source_manifest.py" snapshot --root "$D" --output "$O" --variant scalar --fi-control > "$O/source_snapshot.log"
docker image inspect vllm/vllm-openai:v0.32.0 --format '{{.Id}}' > "$O/image.txt"
printf 'profile=%s\ncache_family=%s\n' "$PROFILE" "$CACHE_FAMILY" > "$O/config.txt"
flags="-v $O/serving:/bench-results -v $HERE:/opt/task-order:ro -v $D/kernels/decode-route-capture:/opt/qwen-fi-control:ro"
flags="$flags -e PYTHONPATH=/opt/task-order:/opt/qwen-fi-control -e TASK_ORDER_OUT=/bench-results/task_order -e TASK_ORDER_PROFILE=$PROFILE -e VLLM_SERVER_DEV_MODE=1"
flags="$flags -e VLLM_SM120_TASK_ORDER_LONG_FIRST=$LONG_FIRST -v $HERE/envs_task_order.py:$PKG/envs.py:ro"
flags="$flags -e VLLM_CACHE_ROOT=/root/.cache/vllm/task-order-$CACHE_FAMILY"
flags="$flags -e QWEN_FI_IMPORT_JSON=/opt/qwen-fi-control/references/next_cpu_audit_autotune.json -e QWEN_FI_IMPORT_SHA256=$FI_SOURCE_SHA -e QWEN_FI_IMPORT_OUT=/bench-results/fi_control -e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/bench-results/fi_control/cache/autotune_import"
K=$D/kernels/moe-nvfp4
flags="$flags -e VLLM_B12X_FIX_SCALES=1 -e VLLM_B12X_SHARED_WORKSPACE=1 -e VLLM_B12X_SCALAR_INPUT_SCALE=1 -e VLLM_B12X_HYBRID_MIN_TOKENS=1024 -e VLLM_B12X_ACT_SCALES=static"
flags="$flags -v $K/scalar/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:$PKG/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:ro -v $K/patched/vllm/model_executor/layers/fused_moe/oracle/unquantized.py:$PKG/model_executor/layers/fused_moe/oracle/unquantized.py:ro"
NOSPEC=1 CONFIG=next NAME=$NAME PORT=$PORT GPU=$GPU NSYS=1 TASK_TRACE_OUT=$O/traces FLAGS="$flags" \
 bash "$HERE/serve_trace.sh" --max-model-len 8192 --max-num-batched-tokens 8192 --max-num-seqs 32 \
 --cudagraph-capture-sizes 1 2 4 8 16 24 32 48 64 96 128 4096 \
 --moe-backend flashinfer_b12x --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --shutdown-timeout 30 > "$O/start.log" 2>&1
[[ -f $O/serving/task_order/startup.json ]] || exit 6
docker exec "$NAME" python3 /opt/task-order/graceful_shutdown.py discover /bench-results/api_process.json > "$O/api_process_discovery.json"
python3 "$D/kernels/decode-route-capture/qwen_fi_control_plugin.py" "$O/serving/fi_control" --source-sha256 "$FI_SOURCE_SHA" --startup-only > "$O/fi_startup_verify.json"
docker inspect "$NAME" > "$O/container.json"
export BENCH_URL=http://127.0.0.1:$PORT
python3 "$HERE/tasks.py" "$O/tasks.json" --prompt "$K/data/prompt.txt" > "$O/tasks.log" 2>&1
docker exec "$NAME" /nsys/target-linux-x64/nsys start --session=vl -o /out/prefill4096 --force-overwrite=false > "$O/trace_start.log" 2>&1
python3 "$HERE/tasks.py" "$O/trace_request.json" --trace-only --prompt "$K/data/prompt.txt" > "$O/trace_request.log" 2>&1
docker exec "$NAME" /nsys/target-linux-x64/nsys stop --session=vl > "$O/trace_stop.log" 2>&1
docker exec "$NAME" /nsys/target-linux-x64/nsys export --type sqlite -o /out/prefill4096.sqlite /out/prefill4096.nsys-rep > "$O/trace_export.log" 2>&1
python3 "$HERE/check_trace.py" "$O/traces/prefill4096.sqlite" "$PROFILE" > "$O/trace_check.json"
curl -fsS "$BENCH_URL/collective_rpc" -H 'Content-Type: application/json' --data '{"method":"task_order_post_workload_audit","timeout":60,"args":[],"kwargs":{}}' > "$O/post_workload_rpc.json"
python3 "$HERE/verify_post_workload.py" "$O/serving" "$D/kernels/decode-route-capture/qwen_fi_control_plugin.py" "$FI_SOURCE_SHA" > "$O/post_workload_verify.json"
