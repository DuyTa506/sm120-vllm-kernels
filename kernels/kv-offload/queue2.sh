#!/bin/bash
# 64 idle sessions, 780 s window: native offload, baseline; then the correctness/PCIe sweep. One GPU-lock hold each.
L=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
cd "$(dirname "$0")"
IDLE="python3 idle_bench.py results/w780_%s.json --sessions 64 --turns 6 --think 5,30 --stagger 3 --seed 0 --duration 780 --tag %s"
flock $L sg docker -c "OFFLOAD=96 LIMIT=900 ./run_cfg.sh w780_native '$(printf "$IDLE" native native)'"
flock $L sg docker -c "LIMIT=900 ./run_cfg.sh w780_base '$(printf "$IDLE" base base)'"
flock $L sg docker -c "OFFLOAD=96 LIMIT=900 ./run_cfg.sh correct_native 'python3 pcie_correct.py results/pc_native.json --targets 10000,25000,50000,75000,100000,120000 --reps 2 --tag pcn'"
