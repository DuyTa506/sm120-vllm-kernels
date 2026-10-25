#!/bin/bash
# End-to-end check of a MoE backend variant on the real server (CONFIG=optimized + this variant).
# usage (takes the GPU lock itself):  kernels/moe-nvfp4/e2e.sh <tag> <cutlass|b12x_fixed|b12x_vllm> [quick]
# Runs: long_eval (prompt NLL on 6 long docs + 20 needles), MTP acceptance from /metrics, decode c=1/4/16/64,
# prefill TTFT at 8192. Results -> kernels/moe-nvfp4/results/e2e_<tag>.*
set -e
D=$(cd "$(dirname "$0")/../.." && pwd); K=$D/kernels/moe-nvfp4; TAG=$1; VAR=$2; QUICK=$3
LOCK=${GPU_LOCK:-/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock}
PKG=/usr/local/lib/python3.12/dist-packages/vllm
NAME=moe-nvfp4-e2e; PORT=8117
mkdir -p $K/results
case $VAR in
  cutlass) F=""; A="" ;;
  b12x_vllm) F="-v $K/patched/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:$PKG/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:ro -v $K/patched/vllm/model_executor/layers/fused_moe/oracle/unquantized.py:$PKG/model_executor/layers/fused_moe/oracle/unquantized.py:ro -e VLLM_B12X_FIX_SCALES=0 -e VLLM_B12X_NEGFIX_ONLY=1"; A="--moe-backend flashinfer_b12x" ;;
  b12x_fixed) F="-v $K/patched/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:$PKG/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:ro -v $K/patched/vllm/model_executor/layers/fused_moe/oracle/unquantized.py:$PKG/model_executor/layers/fused_moe/oracle/unquantized.py:ro -e VLLM_B12X_FIX_SCALES=1"; A="--moe-backend flashinfer_b12x" ;;
  *) echo "unknown variant"; exit 1 ;;
esac
exec 9>$LOCK; flock 9
cleanup() { sg docker -c "docker rm -f $NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
sg docker -c "CONFIG=optimized NAME=$NAME PORT=$PORT FLAGS='$F' $D/scripts/serve.sh $A" > $K/results/e2e_$TAG.serve.log 2>&1 || { tail -40 $K/results/e2e_$TAG.serve.log; exit 1; }
sg docker -c "docker logs $NAME 2>&1" | grep -E "NvFp4 MoE backend|Using .* backend|autotune" | head -5 > $K/results/e2e_$TAG.backend.txt || true
export BENCH_URL=http://127.0.0.1:$PORT
m() { curl -s $BENCH_URL/metrics | grep -E "^vllm:spec_decode_num_(accepted|draft)_tokens_total|^vllm:spec_decode_num_drafts_total" | awk '{print $1, $2}'; }
python3 -I $D/eval/long_eval.py $K/results/e2e_$TAG.long_eval.json > $K/results/e2e_$TAG.long_eval.log 2>&1
m > $K/results/e2e_$TAG.metrics0.txt
sg docker -c "bash $D/bench/bench_decode.sh $NAME" > $K/results/e2e_$TAG.decode.txt 2>&1
m > $K/results/e2e_$TAG.metrics1.txt
if [ -z "$QUICK" ]; then
  sg docker -c "docker exec $NAME vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --served-model-name bench --dataset-name random --ignore-eos --num-warmups 2 --num-prompts 8 --seed 101 --max-concurrency 1 --random-input-len 8192 --random-output-len 1" 2>&1 | grep -E "Mean TTFT|Median TTFT" > $K/results/e2e_$TAG.prefill8k.txt
fi
cat $K/results/e2e_$TAG.backend.txt; grep -E "^NLL|accuracy" $K/results/e2e_$TAG.long_eval.log; cat $K/results/e2e_$TAG.decode.txt $K/results/e2e_$TAG.metrics0.txt $K/results/e2e_$TAG.metrics1.txt; cat $K/results/e2e_$TAG.prefill8k.txt 2>/dev/null
