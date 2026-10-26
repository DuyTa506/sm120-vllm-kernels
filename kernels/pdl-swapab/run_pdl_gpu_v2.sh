#!/bin/bash
# v2: activation loads made coherent (v1 had a hoisted ld.global.nc hazard in one PDL config) + glue-only config.
D=$(cd "$(dirname "$0")" && pwd)
LOCK=${LOCK:-/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock}
exec flock "$LOCK" bash -c '
cd '"$D"'
B=pdl_bench/pdl_chain_v2
nvidia-smi --query-gpu=name,clocks.sm,power.draw,memory.used --format=csv > results/v2_pdl_gpu_state.txt
timeout 120 $B --E 6 > results/v2_pdl_chain_E6.txt 2>&1
timeout 120 $B --E 6 > results/v2_pdl_chain_E6_repeat.txt 2>&1
timeout 120 $B --E 6 --M 16 > results/v2_pdl_chain_E6_M16.txt 2>&1
for c in 0 4 6 7; do
  timeout 120 nsys profile --cuda-graph-trace=node --force-overwrite true -o results/v2_nsys_pdl_cfg$c --export=sqlite \
    $B --cfg $c --L 6 --reps 5 --samples 2 --rounds 1 > results/v2_nsys_pdl_cfg$c.txt 2>&1
done
'
