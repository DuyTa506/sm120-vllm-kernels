#!/bin/bash
# VERIFIER end-to-end check (takes the GPU lock itself, one server per call, ~10 min hold).
# usage: kernels/moe-nvfp4/verify_e2e.sh <tag> <cutlass|b12x_fixed>
# Records: KV-cache size from the server log, prompt logprobs + greedy text (verify_logprobs.py),
# long_eval NLL/needles, 8k TTFT (16 prompts, 2 seeds), 16k TTFT (8 prompts), decode c=1.
set -e
D=$(cd "$(dirname "$0")/../.." && pwd); K=$D/kernels/moe-nvfp4; TAG=$1; VAR=$2
LOCK=${GPU_LOCK:-/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock}
PKG=/usr/local/lib/python3.12/dist-packages/vllm
NAME=moe-nvfp4-verify; PORT=8127; O=$K/results/verify_$TAG
case $VAR in
  cutlass) F=""; A="" ;;
  b12x_fixed) F="-v $K/patched/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:$PKG/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py:ro -v $K/patched/vllm/model_executor/layers/fused_moe/oracle/unquantized.py:$PKG/model_executor/layers/fused_moe/oracle/unquantized.py:ro -e VLLM_B12X_FIX_SCALES=1"; A="--moe-backend flashinfer_b12x" ;;
  *) echo "unknown variant"; exit 1 ;;
esac
exec 9>$LOCK; flock 9
T0=$(date +%s)
cleanup() { sg docker -c "docker logs $NAME" > $O.docker.log 2>&1 || true; sg docker -c "docker rm -f $NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
sg docker -c "CONFIG=optimized NAME=$NAME PORT=$PORT FLAGS='$F' $D/scripts/serve.sh $A" > $O.serve.log 2>&1 || { sg docker -c "docker logs $NAME" > $O.docker.log 2>&1; grep -E "Error|error|memory" $O.docker.log | tail -20; exit 1; }
nvidia-smi --query-gpu=memory.used --format=csv,noheader > $O.gpumem.txt
echo "server up after $(( $(date +%s) - T0 )) s" > $O.summary.txt
sg docker -c "docker logs $NAME 2>&1" | grep -E "NvFp4 MoE backend|KV cache|Available KV|Model loading took|memory profiling|CUDA graph|Graph capturing finished" >> $O.summary.txt || true
export BENCH_URL=http://127.0.0.1:$PORT
python3 -I $K/verify_logprobs.py dump $O.logprobs.json >> $O.summary.txt 2>&1
if [ -z "$TTFT_ONLY" ]; then
python3 -I $D/eval/long_eval.py $O.long_eval.json > $O.long_eval.log 2>&1
grep -E "^NLL|accuracy" $O.long_eval.log >> $O.summary.txt
fi
b() { sg docker -c "docker exec $NAME vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --served-model-name bench --dataset-name random --ignore-eos $*" 2>&1; }
for s in 101 202 ${TTFT_ONLY:+303 404}; do
  echo "8k TTFT seed $s: $(b --num-warmups 2 --num-prompts 16 --seed $s --max-concurrency 1 --random-input-len 8192 --random-output-len 1 | grep -E 'Mean TTFT|Median TTFT|P99 TTFT' | tr -s ' ' | tr '\n' ' ')" >> $O.summary.txt
done
echo "16k TTFT: $(b --num-warmups 1 --num-prompts ${N16:-8} --seed 303 --max-concurrency 1 --random-input-len 16384 --random-output-len 1 | grep -E 'Mean TTFT|Median TTFT' | tr -s ' ' | tr '\n' ' ')" >> $O.summary.txt
echo "decode c=1: $(b --num-warmups 2 --num-prompts 8 --seed 501 --max-concurrency 1 --random-input-len 128 --random-output-len 512 | grep -E 'Output token throughput|Median ITL' | tr -s ' ' | tr '\n' ' ')" >> $O.summary.txt
curl -s $BENCH_URL/metrics | grep -E "^vllm:spec_decode_num_(accepted|draft)_tokens_total" | awk '{print $1, $2}' >> $O.summary.txt
echo "lock held $(( $(date +%s) - T0 )) s" >> $O.summary.txt
cat $O.summary.txt
