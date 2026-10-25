#!/bin/bash
# Variants (short restore/correctness tests): native+prefix-match-unit 16, SimpleCPUOffloadConnector, LMCache MP.
L=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
cd "$(dirname "$0")"
while pgrep -f queue2.sh >/dev/null; do sleep 30; done
PC="python3 pcie_correct.py results/pc_%s.json --targets 25000,100000 --reps 2 --tag %s"
flock $L sg docker -c "OFFLOAD=96 EXTRA='--prefix-match-unit 16' LIMIT=400 ./run_cfg.sh pc_native_pmu16 '$(printf "$PC" native_pmu16 pmu)'; SIMPLE=1 OFFLOAD=96 LIMIT=400 ./run_cfg.sh pc_simple '$(printf "$PC" simple smp)'"
flock $L sg docker -c "KVT_FILE=lmcache_mp.json LIMIT=400 ./run_cfg.sh pc_lmcache '$(printf "$PC" lmcache lmc)'"
