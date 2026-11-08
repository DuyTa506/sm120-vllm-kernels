#!/usr/bin/env bash
# One paired-profile serving diagnostic; no automatic next profile or agentic.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd);D=$(cd "$HERE/../../.." && pwd)
TAG=${1:?unique tag};PROFILE=${2:?off/on};CACHE_FAMILY=${CACHE_FAMILY:?paired family}
[[ $TAG =~ ^[a-zA-Z0-9_-]+$ && $CACHE_FAMILY =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
case $PROFILE in off) ENABLE=0;;on) ENABLE=1;;*)exit 2;;esac
O=$HERE/results/$TAG;NAME=fi-direct-$TAG;PORT=${PORT:-8136};GPU=${GPU:-0}
[[ ! -e $O ]] || exit 2
LOCK=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
exec 9>"$LOCK";flock -n 9 || exit 3
compute=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader)
[[ -z $compute ]] || exit 3
if docker container inspect "$NAME" >/dev/null 2>&1;then exit 3;fi
mkdir -p "$O/serving" "$O/traces"
FI_SHA=7b5146f9b3c1e7cc09586c2087988ea8aaf9da1d86da036539a32dec4b335ec3
cleanup(){
 rc=$?;trap - EXIT;set +e
 docker stop -t 45 "$NAME" > "$O/stop.log" 2>&1
 docker logs "$NAME" > "$O/docker.log" 2>&1
 docker inspect "$NAME" > "$O/container_terminal.json" 2>&1
 docker rm "$NAME" > "$O/remove.log" 2>&1
 docker ps -a --filter "name=^/${NAME}$" --format '{{.ID}}' > "$O/container_after.txt" 2>&1
 if [[ $? -ne 0 || -s $O/container_after.txt ]];then rc=8;fi
 nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$O/gpu_after.csv" 2>&1
 if [[ $? -ne 0 || -s $O/gpu_after.csv ]];then rc=8;fi
 python3 "$D/kernels/decode-route-capture/qwen_fi_control_plugin.py" "$O/serving/fi_control" --source-sha256 "$FI_SHA" > "$O/fi_legacy_shutdown_verify.json" 2> "$O/fi_legacy_shutdown_verify.log"
 echo "$?" > "$O/fi_legacy_shutdown_exit.txt"
 python3 "$HERE/verify_post_workload.py" "$O" "$D/kernels/decode-route-capture/qwen_fi_control_plugin.py" > "$O/post_workload_verify_cleanup.json" 2> "$O/post_workload_verify_cleanup.log" || rc=5
 python3 "$HERE/../provenance.py" "$HERE" "$O/experiment_sources.json" --verify > "$O/source_verify.log" 2>&1 || rc=4
 python3 -I "$D/bench/source_manifest.py" verify --root "$D" --output "$O" > "$O/parent_source_verify.log" 2>&1 || rc=4
 echo "$rc" > "$O/exit_code.txt";exit "$rc"
}
trap cleanup EXIT
python3 "$HERE/../provenance.py" "$HERE" "$O/experiment_sources.json"
python3 -I "$D/bench/source_manifest.py" snapshot --root "$D" --output "$O" --variant scalar --fi-control > "$O/parent_snapshot.log"
docker image inspect dg-sm120-env:1 --format '{{.Id}}' > "$O/image.txt"
PKG=/usr/local/lib/python3.12/dist-packages/vllm
flags="-v $O/serving:/bench-results -v $HERE:/opt/fi-direct:ro -v $D/kernels/decode-route-capture:/opt/qwen-fi-control:ro"
flags="$flags -e PYTHONPATH=/opt/fi-direct:/opt/qwen-fi-control -e FI_DIRECT_PROFILE=$PROFILE -e FI_DIRECT_RUNTIME_OUT=/bench-results/runtime"
flags="$flags -e VLLM_CACHE_ROOT=/root/.cache/vllm/fi-direct-$CACHE_FAMILY -e VLLM_SM120_FI_DIRECT_SMALL_M=$ENABLE -e VLLM_SERVER_DEV_MODE=1"
flags="$flags -e VLLM_B12X_FIX_SCALES=1 -e VLLM_B12X_SHARED_WORKSPACE=1 -e VLLM_B12X_SCALAR_INPUT_SCALE=1 -e VLLM_B12X_HYBRID_MIN_TOKENS=1024 -e VLLM_B12X_ACT_SCALES=static"
flags="$flags -e VLLM_SM120_SKINNY_FP8=0 -e VLLM_SKINNY_BF16_GEMM=0"
flags="$flags -e QWEN_FI_IMPORT_JSON=/opt/qwen-fi-control/references/next_cpu_audit_autotune.json -e QWEN_FI_IMPORT_SHA256=$FI_SHA -e QWEN_FI_IMPORT_OUT=/bench-results/fi_control -e VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/bench-results/fi_control/cache/autotune_import"
while IFS= read -r rel;do flags="$flags -v $HERE/generated/$rel:$PKG/${rel#vllm/}:ro";done < <(python3 - "$HERE/generated/composition.json" <<'PY'
import json,sys
for p in json.load(open(sys.argv[1]))['generated']:print(p)
PY
)
flags="$flags -v $D/kernels/moe-nvfp4/patched/vllm/model_executor/layers/fused_moe/oracle/unquantized.py:$PKG/model_executor/layers/fused_moe/oracle/unquantized.py:ro"
NOSPEC=1 CONFIG=next NAME=$NAME PORT=$PORT GPU=$GPU NSYS=1 FI_DIRECT_TRACE_OUT=$O/traces FLAGS="$flags" \
 bash "$HERE/serve_trace.sh" --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 16384 \
 --cudagraph-capture-sizes 1 2 4 8 16 24 32 --moe-backend flashinfer_b12x \
 --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --shutdown-timeout 30 > "$O/start.log" 2>&1
[[ -f $O/serving/runtime/startup.json ]] || exit 6
docker inspect "$NAME" > "$O/container_start.json"
python3 "$D/kernels/decode-route-capture/qwen_fi_control_plugin.py" "$O/serving/fi_control" --source-sha256 "$FI_SHA" --startup-only > "$O/fi_startup_verify.json"
for task in decode_c1 decode_c4 prefill_2048;do
 case $task in decode_c1) C=1;N=8;I=128;L=128;;decode_c4) C=4;N=16;I=128;L=128;;prefill_2048) C=1;N=8;I=2048;L=1;;esac
 docker exec "$NAME" vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --served-model-name bench --dataset-name random --ignore-eos --num-warmups 8 --num-prompts "$N" --max-concurrency "$C" --random-input-len "$I" --random-output-len "$L" --random-range-ratio 0 --temperature 0 --seed 6421 --save-result --save-detailed --result-dir /bench-results --result-filename "$task.json" > "$O/$task.log" 2>&1
 done
python3 "$HERE/check_results.py" "$O/serving" > "$O/results_check.json"
docker exec "$NAME" /nsys/target-linux-x64/nsys start --session=vl -o /out/live --force-overwrite=false > "$O/trace_start.log" 2>&1
docker exec "$NAME" vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --served-model-name bench --dataset-name random --ignore-eos --num-warmups 0 --num-prompts 8 --max-concurrency 1 --random-input-len 128 --random-output-len 32 --random-range-ratio 0 --temperature 0 --seed 9813 > "$O/trace_requests.log" 2>&1
docker exec "$NAME" /nsys/target-linux-x64/nsys stop --session=vl > "$O/trace_stop.log" 2>&1
docker exec "$NAME" /nsys/target-linux-x64/nsys export --type sqlite -o /out/live.sqlite /out/live.nsys-rep > "$O/trace_export.log" 2>&1
python3 "$HERE/check_trace.py" "$O/traces/live.sqlite" "$PROFILE" > "$O/trace_check.json"
curl -fsS "http://127.0.0.1:$PORT/collective_rpc" -H 'Content-Type: application/json' --data '{"method":"fi_direct_post_workload","timeout":60,"args":[],"kwargs":{}}' > "$O/post_workload_rpc.json"
[[ -f $O/serving/runtime/post_workload.json ]] || exit 6

python3 "$HERE/verify_post_workload.py" "$O" "$D/kernels/decode-route-capture/qwen_fi_control_plugin.py" > "$O/post_workload_verify.json"
