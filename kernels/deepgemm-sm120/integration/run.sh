#!/usr/bin/env bash
# Run in docker group. Exactly one owned server/control per invocation; no agentic stage.
set -euo pipefail
D=$(cd "$(dirname "$0")/../../.." && pwd)
HERE=$D/kernels/deepgemm-sm120/integration
TAG=${1:?unique tag}; PROFILE=${2:?off or on}
[[ $TAG =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
case $PROFILE in off) DG=0;; on) DG=1;; *) exit 2;; esac
CACHE_FAMILY=${CACHE_FAMILY:?paired cache family required}
[[ $CACHE_FAMILY =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
O=$HERE/results/$TAG
NAME=dg-serving-$TAG
PORT=${PORT:-8134}
GPU=${GPU:-0}
SRC=$D/kernels/deepgemm-sm120/fused_src
PKG=/usr/local/lib/python3.12/dist-packages/vllm
FI_SOURCE_SHA=7b5146f9b3c1e7cc09586c2087988ea8aaf9da1d86da036539a32dec4b335ec3
LOCK=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
[[ ! -e $O ]] || exit 2
exec 9>"$LOCK"; flock -n 9 || exit 3
[[ -z $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader) ]] || exit 3
if docker container inspect "$NAME" >/dev/null 2>&1; then echo 'owned name already exists' >&2; exit 3; fi
mkdir -p "$O/serving" "$O/traces" "$O/dg_cache"
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  set +e
  docker stop -t 45 "$NAME" > "$O/stop.log" 2>&1
  python3 "$D/kernels/decode-route-capture/qwen_fi_control_plugin.py" "$O/serving/fi_control" --source-sha256 "$FI_SOURCE_SHA" > "$O/fi_shutdown_verify.json" 2> "$O/fi_shutdown_verify.log"
  if [[ $? -ne 0 && $rc == 0 ]]; then rc=5; fi
  docker logs "$NAME" > "$O/docker.log" 2>&1
  docker rm -f "$NAME" > "$O/remove_container.log" 2>&1
  docker ps -a --filter "name=^/${NAME}$" --format '{{.ID}} {{.Names}}' > "$O/container_after.txt" 2>&1
  if [[ $? -ne 0 || -s $O/container_after.txt ]]; then rc=8; fi
  nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$O/gpu_after.csv" 2>&1
  if [[ $? -ne 0 || -s $O/gpu_after.csv ]]; then rc=8; fi
  python3 "$HERE/provenance.py" "$HERE" "$O/experiment_sources.json" --verify > "$O/experiment_source_verify.log" 2>&1 || rc=4
  python3 "$HERE/source_fingerprint.py" "$SRC" > "$O/dg_source_after.json"
  cmp -s "$O/dg_source.json" "$O/dg_source_after.json" || rc=4
  python3 -I "$D/bench/source_manifest.py" verify --root "$D" --output "$O" > "$O/source_verify.log" 2>&1 || rc=4
  printf '%s\n' "$rc" > "$O/exit_code.txt"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python3 "$HERE/provenance.py" "$HERE" "$O/experiment_sources.json"
python3 "$HERE/source_fingerprint.py" "$SRC" > "$O/dg_source.json"
python3 -I "$D/bench/source_manifest.py" snapshot --root "$D" --output "$O" --variant scalar --fi-control > "$O/source_snapshot.log"
docker image inspect dg-sm120-env:1 --format '{{.Id}}' > "$O/image.txt"
nvidia-smi -i "$GPU" --query-gpu=name,uuid,driver_version,memory.total,memory.used,clocks.sm,power.limit --format=csv > "$O/gpu_before.csv"
printf 'profile=%s\ncache_family=%s\n' "$PROFILE" "$CACHE_FAMILY" > "$O/config.txt"
flags="-v $O/serving:/bench-results -v $SRC:/work/src:ro -v $O/dg_cache:/dg-cache -e DG_JIT_CACHE_DIR=/dg-cache"
flags="$flags -e VLLM_CACHE_ROOT=/root/.cache/vllm/dg-integration-$CACHE_FAMILY"
flags="$flags -e CUDA_HOME=/usr/local/cuda-13.0 -e CPLUS_INCLUDE_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/include -e LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib"
flags="$flags -v $HERE:/opt/dg-integration:ro -v $D/kernels/decode-route-capture:/opt/qwen-fi-control:ro"
flags="$flags -e PYTHONPATH=/work/src:/opt/dg-integration:/opt/qwen-fi-control -e DG_RUNTIME_OUT=/bench-results/dg_runtime -e DG_PROFILE=$PROFILE"
flags="$flags -e VLLM_SERVER_DEV_MODE=1 -e VLLM_SM120_DG_ROWWISE_FP8=$DG -e VLLM_SM120_SKINNY_FP8=0 -e VLLM_SKINNY_BF16_GEMM=0 -e VLLM_SM120_SKINNY_FP8_MAX_M=64"
flags="$flags -e QWEN_FI_IMPORT_JSON=/opt/qwen-fi-control/references/next_cpu_audit_autotune.json -e QWEN_FI_IMPORT_SHA256=$FI_SOURCE_SHA -e QWEN_FI_IMPORT_OUT=/bench-results/fi_control"
flags="$flags -e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/bench-results/fi_control/cache/autotune_import"
while IFS= read -r relative; do flags="$flags -v $HERE/generated/$relative:$PKG/${relative#vllm/}:ro"; done < <(python3 - "$HERE/generated/composition.json" <<'PY'
import json,sys
for name in json.load(open(sys.argv[1]))['generated_sha256']:print(name)
PY
)
K=$D/kernels/moe-nvfp4
flags="$flags -e VLLM_B12X_FIX_SCALES=1 -e VLLM_B12X_SHARED_WORKSPACE=1 -e VLLM_B12X_SCALAR_INPUT_SCALE=1"
flags="$flags -v $K/scalar/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:$PKG/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:ro"
flags="$flags -v $K/patched/vllm/model_executor/layers/fused_moe/oracle/unquantized.py:$PKG/model_executor/layers/fused_moe/oracle/unquantized.py:ro"
NOSPEC=1 CONFIG=next NAME=$NAME PORT=$PORT GPU=$GPU NSYS=1 DG_TRACE_OUT=$O/traces FLAGS="$flags" \
 bash "$HERE/serve_trace.sh" --max-model-len 8192 --max-num-batched-tokens 2048 --max-num-seqs 32 \
 --cudagraph-capture-sizes 1 2 4 8 16 24 32 33 48 63 64 65 80 96 112 128 \
 --moe-backend flashinfer_b12x --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --shutdown-timeout 30 > "$O/start.log" 2>&1
[[ -f $O/serving/dg_runtime/startup.json ]] || { echo 'Missing runtime eligibility guard'; exit 6; }
python3 "$D/kernels/decode-route-capture/qwen_fi_control_plugin.py" "$O/serving/fi_control" --source-sha256 "$FI_SOURCE_SHA" --startup-only > "$O/fi_startup_verify.json"
docker inspect "$NAME" > "$O/container.json"
export BENCH_URL=http://127.0.0.1:$PORT
curl -fsS "$BENCH_URL/metrics" > "$O/metrics_before.txt"
# Focused task micro shapes mirror serving_matrix decode-c16 and prefill-2048.
for mode in prefill decode; do
 if [[ $mode == prefill ]]; then INPUT=2048; OUTPUT=1; CONCURRENCY=1; PROMPTS=8; else INPUT=128; OUTPUT=128; CONCURRENCY=16; PROMPTS=32; fi
 docker exec "$NAME" vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --served-model-name bench --dataset-name random --ignore-eos --num-warmups 16 --num-prompts "$PROMPTS" --max-concurrency "$CONCURRENCY" --random-input-len "$INPUT" --random-output-len "$OUTPUT" --temperature 0 --seed 6421 --save-result --save-detailed --result-dir /bench-results --result-filename "$mode.json" > "$O/$mode.log" 2>&1
 done
python3 "$HERE/small_nll.py" "$O/small_nll.json" "$K/data/prompt.txt" > "$O/small_nll.log" 2>&1
# Live graph trace follows warmup; do not mistake eager preparation for execution.
docker exec "$NAME" /nsys/target-linux-x64/nsys start --session=vl -o /out/c16 --force-overwrite=false > "$O/trace_start.log" 2>&1
docker exec "$NAME" vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --served-model-name bench --dataset-name random --ignore-eos --num-warmups 0 --num-prompts 16 --max-concurrency 16 --random-input-len 128 --random-output-len 32 --temperature 0 --seed 9813 > "$O/trace_requests.log" 2>&1
docker exec "$NAME" /nsys/target-linux-x64/nsys stop --session=vl > "$O/trace_stop.log" 2>&1
docker exec "$NAME" /nsys/target-linux-x64/nsys export --type sqlite -o /out/c16.sqlite /out/c16.nsys-rep > "$O/trace_export.log" 2>&1
python3 "$HERE/check_trace.py" "$O/traces/c16.sqlite" "$PROFILE" > "$O/trace_check.json"
curl -fsS "$BENCH_URL/metrics" > "$O/metrics_after.txt"
# Explicit worker-alive checkpoint after all requests/tracing. Shutdown remains separately verified.
curl -fsS "$BENCH_URL/collective_rpc" -H 'Content-Type: application/json' \
 --data '{"method":"dg_post_workload_audit","timeout":60,"args":[],"kwargs":{}}' > "$O/post_workload_rpc.json"
python3 "$HERE/verify_post_workload.py" "$O/serving" "$D/kernels/decode-route-capture/qwen_fi_control_plugin.py" "$FI_SOURCE_SHA" > "$O/post_workload_verify.json"
