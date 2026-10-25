#!/bin/bash
# Main idle-session experiment: baseline and native CPU offloading, one GPU-lock hold each.
L=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
cd "$(dirname "$0")"
IDLE="python3 idle_bench.py results/idle64_%s.json --sessions 64 --turns 6 --think 5,30 --stagger 3 --seed 0 --tag %s"
flock $L sg docker -c "LIMIT=1000 ./run_cfg.sh idle64_base '$(printf "$IDLE" base base)'"
flock $L sg docker -c "OFFLOAD=96 LIMIT=1000 ./run_cfg.sh idle64_native '$(printf "$IDLE" native native)'"
