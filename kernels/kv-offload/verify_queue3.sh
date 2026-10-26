#!/bin/bash
# Verifier hold: correctness probes under load WITH real CPU restores and CPU-tier eviction.
# (The 12 GiB run gave 0 external hits: a write-through CPU tier smaller than GPU KV holds nothing the GPU lacks.)
# 64 GiB CPU (~6M attention tokens) vs a 64-session background (6.1M) + 16 probes -> CPU restores and CPU LRU eviction.
L=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
cd "$(dirname "$0")"
while pgrep -f "verify_queue2?.sh" >/dev/null; do sleep 30; done
flock $L sg docker -c "OFFLOAD=64 LIMIT=840 ./verify_run.sh vload64 'python3 idle_bench.py verify_results/load64_bg.json --sessions 64 --turns 6 --think 5,30 --stagger 3 --seed 4 --duration 720 --tag vbg64 > verify_logs/vload64.bg.log 2>&1 & python3 verify_load.py verify_results/load64_probes.json --probes 16 --stagger 15 --sleep 60,200 --len 20000,60000 --tag vl64; wait; tail -3 verify_logs/vload64.bg.log'"
echo "QUEUE3 DONE $(date +%T)"
