#!/bin/bash
# Verifier: does VLLM_SKINNY_BF16_GEMM=0 really mean "off" once the patch files are installed?
# One GPU-lock hold: (1) patched + flag=1 server (warms the shared AOT compile cache), one greedy c=1 bench;
# (2) patched + flag=0 server on the same cache, full engine log kept, one greedy c=1 bench if it starts.
D=$(cd "$(dirname "$0")" && pwd)
LOCK=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
PKG=/usr/local/lib/python3.12/dist-packages/vllm
NAME=decode-roofline-verify
MNT="-v $D/vllm/model_executor/layers/utils.py:$PKG/model_executor/layers/utils.py:ro -v $D/vllm/model_executor/layers/skinny_bf16_gemm.py:$PKG/model_executor/layers/skinny_bf16_gemm.py:ro"
cat > $D/results/.verify_off_inner.sh <<EOF
B="docker exec $NAME vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --served-model-name bench --dataset-name random --ignore-eos --random-input-len 128 --random-output-len 512"
for FL in 1 0; do
  echo "=== patched files, VLLM_SKINNY_BF16_GEMM=\$FL"
  FLAGS="-e VLLM_SKINNY_BF16_GEMM=\$FL $MNT" CONFIG=optimized NAME=$NAME PORT=8106 $D/serve_nsys.sh > /dev/null 2>&1
  docker logs $NAME > $D/results/verify_flag\${FL}_server.log 2>&1
  if curl -sf http://127.0.0.1:8106/health >/dev/null; then
    \$B --num-warmups 2 --num-prompts 4 --max-concurrency 1 --seed 1 >/dev/null 2>&1
    echo "greedy c=1 seed 501: \$(\$B --temperature 0 --num-warmups 1 --num-prompts 16 --seed 501 --max-concurrency 1 2>&1 | grep -E 'Output token throughput|Median ITL|Acceptance length' | tr -s ' ' | tr '\n' ' ')"
  else
    echo "server failed to start"
  fi
  grep -E "SKINNY|Directly load|Compiling model again|Error|error" $D/results/verify_flag\${FL}_server.log | grep -v "Initializing a V1" | cut -c1-400 | head -20
  docker rm -f $NAME >/dev/null 2>&1
done
EOF
flock $LOCK sg docker -c "bash $D/results/.verify_off_inner.sh 2>&1 | tee $D/results/verify_flag_off.txt; docker rm -f $NAME >/dev/null 2>&1"
rm -f $D/results/.verify_off_inner.sh
