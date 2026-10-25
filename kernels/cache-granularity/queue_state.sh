#!/bin/bash
# usage: LOCK=<gpu.lock> queue_state.sh "label|prod|opt|vllm args" ...   (each job in its own lock hold)
H=$(cd "$(dirname "$0")" && pwd)
for job in "$@"; do
  IFS='|' read -r label kind args <<< "$job"
  echo "### waiting for lock: $label ($(date +%T))"
  flock $LOCK sg docker -c "$H/run_state.sh $label $kind $args"
  echo "### finished $label ($(date +%T))"
done
