# Stage 6: calibration of the correctness check. Server runs the FAULT=1 worker (fault/mamba_hybrid.py):
# every session snapshot is taken one token off. Same cases as stage 3; a check with power must show the
# hot-vs-cold divergence clearly above the stage-3 (correct) level.
set -x
S=../../results/agentic/trajectories_short_production.json
L=../../results/agentic/trajectories_long_production.json
O=results/s6_fault; mkdir -p $O
python3 hot_vs_cold.py $S $O/hvc_short.json --sessions 3,4,5,6,7 --turns 13 --cmp-tokens 24
python3 hot_vs_cold.py $L $O/hvc_long.json --sessions 1 --turns 9 --cmp-tokens 24
