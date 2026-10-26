#!/bin/bash
# Verifier queue: each job = one GPU-lock hold. args: "label|config|steps|vllm args" ...
H=$(cd "$(dirname "$0")" && pwd)
LOCK=${LOCK:?set LOCK}
for job in "$@"; do
  IFS='|' read -r label cfg steps args <<< "$job"
  echo "### waiting for lock: $label ($(date))"
  STEPS="$steps" flock $LOCK sg docker -c "STEPS='$steps' $H/verify_run.sh $label $cfg $args"
  echo "### released: $label ($(date))"
done
