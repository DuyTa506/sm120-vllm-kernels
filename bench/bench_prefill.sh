#!/bin/bash
# usage: bench_prefill.sh <container> ; prefill TTFT (median/mean) for several lengths, fresh seeds each so no prefix-cache hits
C=${1:-bench-vllm}
M=unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
SEED0=${SEED0:-100}
b() { docker exec $C vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model $M --served-model-name bench --dataset-name random --ignore-eos "$@" 2>&1; }
i=0
for L in 2048 4096 5000 8192 16384 32768; do
  i=$((i+1))
  r=$(b --num-warmups 2 --num-prompts 8 --seed $((SEED0+i)) --max-concurrency 1 --random-input-len $L --random-output-len 1 | grep -E 'Mean TTFT|Median TTFT' | tr -s ' ' | tr '\n' ' ')
  echo "prefill $L: $r"
done
