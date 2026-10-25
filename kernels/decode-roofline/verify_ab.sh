#!/bin/bash
# Verifier's untraced end-to-end A/B. One GPU-lock hold per invocation, container removed at the end.
# usage: verify_ab.sh <tag> base|skinny|patched_off
#   base        image vLLM + patches 01/02 (CONFIG=optimized)
#   skinny      + patched utils.py/skinny_bf16_gemm.py, VLLM_SKINNY_BF16_GEMM=1
#   patched_off + patched files mounted, VLLM_SKINNY_BF16_GEMM=0 (tests the "default off" claim with a
#               warm torch.compile cache: does the skinny Triton kernel still run?)
# Benches: greedy c=1 x3 seeds, default sampling c=1 x2, greedy c=2 (M=6), greedy c=4 (M=12).
D=$(cd "$(dirname "$0")" && pwd)
TAG=$1; VARIANT=${2:-base}; QUICK=${QUICK:-0}
LOCK=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
PKG=/usr/local/lib/python3.12/dist-packages/vllm
NAME=decode-roofline-verify
MNT="-v $D/vllm/model_executor/layers/utils.py:$PKG/model_executor/layers/utils.py:ro -v $D/vllm/model_executor/layers/skinny_bf16_gemm.py:$PKG/model_executor/layers/skinny_bf16_gemm.py:ro"
case $VARIANT in
  base) EXTRA="" ;;
  skinny) EXTRA="-e VLLM_SKINNY_BF16_GEMM=1 $MNT" ;;
  patched_off) EXTRA="-e VLLM_SKINNY_BF16_GEMM=0 $MNT" ;;
esac
mkdir -p $D/results
OUTF=$D/results/verify_${TAG}.txt
cat > $D/results/.verify_inner_$TAG.sh <<EOF
set -e
T0=\$(date +%s)
FLAGS='$EXTRA' CONFIG=optimized NAME=$NAME PORT=8106 $D/serve_nsys.sh
echo "server up after \$((\$(date +%s)-T0)) s"
B="docker exec $NAME vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --served-model-name bench --dataset-name random --ignore-eos --random-input-len 128 --random-output-len 512"
G='Output token throughput|Median TPOT|Median ITL|Acceptance length'
\$B --num-warmups 2 --num-prompts 4 --max-concurrency 1 --seed 1 >/dev/null 2>&1
if [ "$QUICK" = 1 ]; then
  echo "greedy c=1 seed 501: \$(\$B --temperature 0 --num-warmups 1 --num-prompts 16 --seed 501 --max-concurrency 1 2>&1 | grep -E "\$G" | tr -s ' ' | tr '\n' ' ')"
else
  for s in 501 601 701; do
    echo "greedy c=1 seed \$s: \$(\$B --temperature 0 --num-warmups 1 --num-prompts 16 --seed \$s --max-concurrency 1 2>&1 | grep -E "\$G" | tr -s ' ' | tr '\n' ' ')"
  done
  for s in 501 601; do
    echo "sampling c=1 seed \$s: \$(\$B --num-warmups 1 --num-prompts 16 --seed \$s --max-concurrency 1 2>&1 | grep -E "\$G" | tr -s ' ' | tr '\n' ' ')"
  done
  echo "greedy c=2: \$(\$B --temperature 0 --num-warmups 2 --num-prompts 12 --seed 502 --max-concurrency 2 2>&1 | grep -E "\$G" | tr -s ' ' | tr '\n' ' ')"
  echo "greedy c=4: \$(\$B --temperature 0 --num-warmups 2 --num-prompts 24 --seed 504 --max-concurrency 4 2>&1 | grep -E "\$G" | tr -s ' ' | tr '\n' ' ')"
fi
echo "skinny triton kernel compiled in this container: \$(docker exec $NAME bash -c 'grep -rl _skinny_bf16_gemm_kernel /root/.triton /root/.cache 2>/dev/null | grep -v huggingface | head -1' || true)"
echo "total hold \$((\$(date +%s)-T0)) s"
EOF
flock $LOCK sg docker -c "bash $D/results/.verify_inner_$TAG.sh 2>&1 | tee $OUTF; docker logs $NAME 2>&1 | grep -iE 'SKINNY|Traceback|torch_compile_cache|Directly load|Compiling a graph|cache_dir' | head -8 | tee -a $OUTF; docker rm -f $NAME >/dev/null"
rm -f $D/results/.verify_inner_$TAG.sh
