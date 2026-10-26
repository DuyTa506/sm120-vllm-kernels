#!/bin/bash
# Correctness (bitwise vs the production op) + microbenchmark, one GPU-lock hold.
# usage: run_kernel_tests.sh [tag]      results -> results/kernel_<tag>.{correctness,bench}.txt
set -e
D=$(cd "$(dirname "$0")" && pwd)
TAG=${1:-run}
LOCK=${LOCK:-/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock}
mkdir -p $D/results
exec 9>>"$LOCK"
flock 9
echo "lock acquired $(date +%T)"
$D/run_in_container.sh --gpu tests/test_correctness.py 2>&1 | grep -v Warn | tee $D/results/kernel_$TAG.correctness.txt || true
$D/run_in_container.sh --gpu bench.py 2>&1 | grep -v Warn | tee $D/results/kernel_$TAG.bench.txt
echo "done $(date +%T)"
