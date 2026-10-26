cd /work
export PYTHONPATH=/work/src:/work/src/tests
timeout 600 python3 -u bench_ours_kernels.py 2>&1 | grep -v "Making TMA\|autotuner\|AutoTuner" | tail -120
timeout 600 python3 -u bench_dg_kernels.py 2>&1 | grep -v "Making TMA" | tail -40
