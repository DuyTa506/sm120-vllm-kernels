#!/bin/bash
# One GPU-lock hold, no nsys: start the optimized server (base or skinny), run decode benches at
# c=1/4/64 (greedy temperature 0, so both sides do comparable work) plus the default-sampling c=1
# bench, and the greedy quality probe twice (run-to-run determinism check). Removes the container.
# usage: kernels/decode-roofline/run_ab.sh <tag> base|skinny
D=$(cd "$(dirname "$0")" && pwd)
TAG=$1; VARIANT=${2:-base}
LOCK=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
PKG=/usr/local/lib/python3.12/dist-packages/vllm
EXTRA=""
if [ "$VARIANT" = skinny ]; then
  EXTRA="-e VLLM_SKINNY_BF16_GEMM=1 -v $D/vllm/model_executor/layers/utils.py:$PKG/model_executor/layers/utils.py:ro -v $D/vllm/model_executor/layers/skinny_bf16_gemm.py:$PKG/model_executor/layers/skinny_bf16_gemm.py:ro"
fi
mkdir -p $D/results
cat > $D/results/.ab_inner.sh <<EOF
set -e
FLAGS='$EXTRA' CONFIG=optimized NAME=decode-agent PORT=8105 $D/serve_nsys.sh
B="docker exec decode-agent vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --served-model-name bench --dataset-name random --ignore-eos --random-input-len 128 --random-output-len 512"
\$B --num-warmups 2 --num-prompts 4 --max-concurrency 1 --seed 1 >/dev/null 2>&1
for c in 1 4 64; do
  n=\$((c*6)); [ \$c -eq 1 ] && n=16
  echo "greedy c=\$c: \$(\$B --temperature 0 --num-warmups 2 --num-prompts \$n --seed \$((500+c)) --max-concurrency \$c 2>&1 | grep -E 'Output token throughput|Median TPOT|Median ITL|Acceptance length' | tr -s ' ' | tr '\n' ' ')"
done
echo "default-sampling c=1: \$(\$B --num-warmups 2 --num-prompts 16 --seed 501 --max-concurrency 1 2>&1 | grep -E 'Output token throughput|Median TPOT|Median ITL|Acceptance length' | tr -s ' ' | tr '\n' ' ')"
BENCH_URL=http://127.0.0.1:8105 python3 -I $D/../../eval/quality_probe.py $D/results/${TAG}_probeA.json
BENCH_URL=http://127.0.0.1:8105 python3 -I $D/../../eval/quality_probe.py $D/results/${TAG}_probeB.json
EOF
flock $LOCK sg docker -c "bash $D/results/.ab_inner.sh 2>&1 | tee $D/results/${TAG}_ab.txt; docker logs decode-agent 2>&1 | grep -iE 'SKINNY|Traceback' | head -3; docker rm -f decode-agent >/dev/null"
rm -f $D/results/.ab_inner.sh
