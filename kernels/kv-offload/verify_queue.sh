#!/bin/bash
# Verifier job list: one GPU-lock hold per configuration.
L=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
cd "$(dirname "$0")"
mkdir -p verify_logs verify_results
# H1: sequential correctness incl. divergent / mixed GPU+CPU / exact-retry restores, spec-decode acceptance
flock $L sg docker -c "OFFLOAD=96 LIMIT=900 ./verify_run.sh vseq 'python3 verify_seq.py verify_results/seq.json --targets 30000,60000,100000 --reps 2'"
# H2: correctness under concurrent load with a SMALL CPU tier (12 GiB) so CPU LRU eviction happens
flock $L sg docker -c "OFFLOAD=12 LIMIT=840 ./verify_run.sh vload 'python3 idle_bench.py verify_results/load_bg.json --sessions 36 --turns 6 --think 5,30 --stagger 4 --seed 3 --duration 600 --tag vbg > verify_logs/vload.bg.log 2>&1 & python3 verify_load.py verify_results/load_probes.json --probes 16 --stagger 12 --sleep 40,150; wait; tail -3 verify_logs/vload.bg.log'"
# H3/H4: independent re-run of the 64-session idle benchmark with a different seed (2), 700 s window
IDLE="python3 idle_bench.py verify_results/w700s2_%s.json --sessions 64 --turns 6 --think 5,30 --stagger 3 --seed 2 --duration 700 --tag v%s"
flock $L sg docker -c "OFFLOAD=96 LIMIT=800 ./verify_run.sh vw_native '$(printf "$IDLE" native native)'"
flock $L sg docker -c "LIMIT=800 ./verify_run.sh vw_base '$(printf "$IDLE" base base)'"
echo "QUEUE DONE $(date +%T)"
