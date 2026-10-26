#!/bin/bash
# LMCache MP without MTP (NOSPEC=1; block size is 2096 without the MTP layer): is the wrong state MTP-specific?
L=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
cd "$(dirname "$0")"

flock $L sg docker -c "NOSPEC=1 KVT_FILE=lmcache_mp_nospec.json LIMIT=400 ./run_cfg.sh pc_lmcache_nospec 'python3 pcie_correct.py results/pc_lmcache_nospec.json --targets 25000,100000 --reps 2 --tag lmn'"
