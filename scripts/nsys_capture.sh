#!/bin/bash
# Record 4 single-request prefills on a server started with NSYS=1, then report the GPU span per request.
# usage: scripts/nsys_capture.sh <name> [input_len] [container]
#   add NSYS_OPTS=--cuda-graph-trace=node to serve.sh's launch for per-kernel tables (bench/nsys_kernels.py)
D=$(cd "$(dirname "$0")/.." && pwd)
NAME=$1; L=${2:-8192}; C=${3:-bench-vllm}; N=/nsys/target-linux-x64/nsys
B="docker exec $C vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --served-model-name bench --dataset-name random --ignore-eos --random-input-len $L --random-output-len 1 --max-concurrency 1"
$B --num-warmups 3 --num-prompts 2 --seed 1 >/dev/null 2>&1
docker exec $C $N start --session=vl -o /out/$NAME --force-overwrite=true >/dev/null
$B --num-warmups 0 --num-prompts 4 --seed 777 2>&1 | grep -E "Mean TTFT"
docker exec $C $N stop --session=vl >/dev/null
cd $D/results/nsys/raw && nsys export --type sqlite --force-overwrite=true -o $NAME.sqlite $NAME.nsys-rep >/dev/null
python3 -I $D/bench/nsys_analyze.py $D/results/nsys/raw/$NAME.sqlite
