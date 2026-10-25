#!/bin/bash
# Throughput sweep for every config, one server at a time. usage: ./run_sweep_all.sh [configs...]
D=$(cd "$(dirname "$0")" && pwd)
mkdir -p $D/results/throughput
for c in ${@:-production tuned optimized}; do
  CONFIG=$c $D/scripts/serve.sh || exit 1
  $D/bench/throughput_sweep.sh $D/results/throughput/$c.tsv
done
docker rm -f bench-vllm >/dev/null
