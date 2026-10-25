#!/bin/bash
# Record decode traces (c=1 and c=64, random 128 in / 256 out) on a server started with
# serve_nsys.sh (NSYS=1), plus an untraced-window decode bench and a greedy quality probe.
# Run under the GPU lock. Writes traces/<tag>_c{1,64}.{nsys-rep,sqlite}, results/<tag>_*.
# usage: capture.sh <tag> [container] [port]
D=$(cd "$(dirname "$0")" && pwd)
TAG=$1; C=${2:-decode-agent}; PORT=${3:-8105}; N=/nsys/target-linux-x64/nsys
mkdir -p $D/results
B="docker exec $C vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --served-model-name bench --dataset-name random --ignore-eos --random-input-len 128"
$B --random-output-len 256 --num-warmups 2 --num-prompts 4 --max-concurrency 1 --seed 1 >/dev/null 2>&1
# decode throughput outside the nsys window (same as bench/bench_decode.sh c=1 and c=64)
for c in 1 64; do
  n=$((c*6)); [ $c -eq 1 ] && n=8
  echo "decode c=$c: $($B --random-output-len 512 --num-warmups 2 --num-prompts $n --seed $((500+c)) --max-concurrency $c 2>&1 | grep -E 'Output token throughput|Median TPOT|Median ITL|Acceptance length' | tr -s ' ' | tr '\n' ' ')"
done | tee $D/results/${TAG}_decode.txt
# greedy quality probe (12 prompts, 160 tokens, temperature 0, logprobs)
BENCH_URL=http://127.0.0.1:$PORT python3 -I $D/../../eval/quality_probe.py $D/results/${TAG}_probe.json
for c in 1 64; do
  n=$c; [ $c -eq 1 ] && n=3
  docker exec $C $N start --session=vl -o /out/${TAG}_c$c --force-overwrite=true >/dev/null
  $B --random-output-len 256 --num-warmups 0 --num-prompts $n --max-concurrency $c --seed $((900+c)) 2>&1 | grep -E "Output token throughput|Median TPOT|Median ITL|Acceptance length" | tr -s ' ' | sed "s/^/nsys c=$c /"
  docker exec $C $N stop --session=vl >/dev/null
done
cd $D/traces && for c in 1 64; do nsys export --type sqlite --force-overwrite=true -o ${TAG}_c$c.sqlite ${TAG}_c$c.nsys-rep >/dev/null 2>&1; done
ls -la $D/traces | grep $TAG
