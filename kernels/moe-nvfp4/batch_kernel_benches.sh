#!/bin/bash
# Runs inside the container (cwd = kernels/moe-nvfp4): all kernel-level benches, logs to results/.
mkdir -p results
python3 numerics_b12x.py 0 9 19 29 39 2>&1 | grep -v Warning | tee results/numerics_b12x.txt
python3 bench_b12x_vs_cutlass.py 19 2>&1 | grep -v Warning | tee results/b12x_vs_cutlass_L19.txt
python3 bench_breakdown.py 19 8192 4096 48 12 3 2>&1 | grep -v Warning | tee results/breakdown_L19.txt
python3 bench_tactics.py 19 8192 4096 1024 192 48 12 3 2>&1 | grep -v Warning | tee results/tactics_L19.txt
