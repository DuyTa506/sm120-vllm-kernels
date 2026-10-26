#!/bin/bash
# Re-run of the windowed baseline (the first attempt found the GPU occupied by another job's server).
L=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
cd "$(dirname "$0")"
flock $L sg docker -c "LIMIT=900 ./run_cfg.sh w780_base 'python3 idle_bench.py results/w780_base.json --sessions 64 --turns 6 --think 5,30 --stagger 3 --seed 0 --duration 780 --tag base'"
