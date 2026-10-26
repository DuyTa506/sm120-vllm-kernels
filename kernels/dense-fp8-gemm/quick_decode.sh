#!/bin/bash
# usage: quick_decode.sh <container> ; c=1 (16 prompts) and c=4 (32 prompts), random 128-in / 512-out, MTP on.
# Same request generator as bench/bench_decode.sh, more prompts for a tighter median.
C=${1:-bench-vllm}
M=unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
b() { docker exec $C vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model $M --served-model-name bench --dataset-name random --ignore-eos "$@" 2>&1; }
for c in 1 4; do
  n=$((c == 1 ? 16 : 32))
  echo "decode c=$c: $(b --num-warmups 2 --num-prompts $n --seed $((700+c)) --max-concurrency $c --random-input-len 128 --random-output-len 512 | grep -E 'Output token throughput|Median TPOT|Median ITL|Mean ITL' | tr -s ' ' | tr '\n' ' ')"
done
