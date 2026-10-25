# Stage 3: correctness, more cases (hot vs cold vs normal prefix hit vs off-by-one negative control).
set -x
S=../../results/agentic/trajectories_short_production.json
L=../../results/agentic/trajectories_long_production.json
O=results/s3; mkdir -p $O
python3 hot_vs_cold.py $S $O/hvc_short.json --sessions 3,4,5,6,7 --turns 13 --cmp-tokens 24
python3 hot_vs_cold.py $L $O/hvc_long.json --sessions 1 --turns 9 --cmp-tokens 24
