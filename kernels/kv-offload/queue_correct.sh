#!/bin/bash
# Correctness + PCIe restore cost sweep on native offloading (one hold).
L=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
cd "$(dirname "$0")"
while pgrep -f queue_main.sh >/dev/null; do sleep 20; done
flock $L sg docker -c "OFFLOAD=96 LIMIT=900 ./run_cfg.sh correct_native 'python3 pcie_correct.py results/pc_native.json --targets 10000,25000,50000,75000,100000,120000 --reps 2 --tag pcn'"
