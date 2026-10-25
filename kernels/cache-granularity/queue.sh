#!/bin/bash
# Run a list of measurement jobs, each in its own hold of the GPU lock (others can interleave between jobs).
# usage: LOCK=<gpu.lock> queue.sh "label|config|steps|vllm args" ...
H=$(cd "$(dirname "$0")" && pwd)
for job in "$@"; do
  IFS='|' read -r label cfg steps args <<< "$job"
  echo "### waiting for lock: $label ($(date +%T))"
  flock $LOCK sg docker -c "STEPS='$steps' $H/run_config.sh $label $cfg $args"
  echo "### finished $label ($(date +%T))"
done
