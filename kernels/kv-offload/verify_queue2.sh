#!/bin/bash
# Verifier follow-up hold: repeated cold / GPU-retry / CPU-retry on the same prompts (noise vs offload effect).
L=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
cd "$(dirname "$0")"
while pgrep -f verify_queue.sh >/dev/null; do sleep 30; done
flock $L sg docker -c "OFFLOAD=96 LIMIT=700 ./verify_run.sh vrep 'python3 verify_repeat.py verify_results/repeat.json --cases 6:22,5:11,7:16,0:40 --n 4'"
echo "QUEUE2 DONE $(date +%T)"
