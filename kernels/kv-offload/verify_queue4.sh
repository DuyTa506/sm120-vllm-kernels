#!/bin/bash
# Verifier holds:
# H6: sizing cliff, clean: same 64-session seed-2 workload as w700s2_{native,base} but a 64 GiB CPU tier, no probes.
# H7: correctness of CPU restores under heavy concurrency: 96 GiB, 48-session background (fits CPU) + 16 smaller probes.
L=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
cd "$(dirname "$0")"
flock $L sg docker -c "OFFLOAD=64 LIMIT=800 ./verify_run.sh vw_off64 'python3 idle_bench.py verify_results/w700s2_off64.json --sessions 64 --turns 6 --think 5,30 --stagger 3 --seed 2 --duration 700 --tag voff64'"
flock $L sg docker -c "OFFLOAD=96 LIMIT=800 ./verify_run.sh vload96 'python3 idle_bench.py verify_results/load96_bg.json --sessions 48 --turns 6 --think 5,30 --stagger 4 --seed 5 --duration 700 --tag vbg96 > verify_logs/vload96.bg.log 2>&1 & python3 verify_load.py verify_results/load96_probes.json --probes 16 --stagger 10 --sleep 40,120 --len 10000,30000 --tag vl96; wait; tail -2 verify_logs/vload96.bg.log'"
echo "QUEUE4 DONE $(date +%T)"
