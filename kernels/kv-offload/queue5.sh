#!/bin/bash
# Moderate load: 48 sessions spread over 240 s, 600 s window; baseline (+ cold-TTFT reference) and native.
L=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
cd "$(dirname "$0")"
while pgrep -f queue3.sh >/dev/null; do sleep 30; done
IDLE="python3 idle_bench.py results/w600s48_%s.json --sessions 48 --turns 6 --think 5,30 --stagger 5 --seed 1 --duration 600 --tag %s48"
flock $L sg docker -c "LIMIT=700 ./run_cfg.sh w600s48_base '$(printf "$IDLE" base base)' 'python3 pcie_correct.py results/pc_base.json --targets 25000,50000,100000 --reps 1 --tag pcb'"
flock $L sg docker -c "OFFLOAD=96 LIMIT=700 ./run_cfg.sh w600s48_native '$(printf "$IDLE" native native)'"
