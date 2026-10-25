#!/bin/bash
# usage: bench_decode.sh <container> ; decode throughput/TPOT at several concurrencies (random 128-in / 512-out, MTP on)
C=${1:-bench-vllm}
M=unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
SEED0=${SEED0:-500}
b() { docker exec $C vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model $M --served-model-name bench --dataset-name random --ignore-eos "$@" 2>&1; }
for c in 1 4 16 64; do
  n=$((c*6)); [ $c -eq 1 ] && n=8
  echo "decode c=$c: $(b --num-warmups 2 --num-prompts $n --seed $((SEED0+c)) --max-concurrency $c --random-input-len 128 --random-output-len 512 | grep -E 'Output token throughput|Median TPOT|Median ITL' | tr -s ' ' | tr '\n' ' ')"
done
