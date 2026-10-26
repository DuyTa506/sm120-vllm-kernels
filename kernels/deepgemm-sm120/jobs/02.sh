# Second lock hold: our FP4 path with cold weights, dense re-run incl. our skinny kernel, DeepGEMM's remaining SM120 tests.
cd /work
export PYTHONPATH=/work/src:/work/src/tests
export DG_PRINT_CONFIGS=1
stage() { echo; echo "=== STAGE $1 start $(date +%T)"; }
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,memory.used --format=csv
stage ours_cold
timeout 600 python3 -u bench_ours_cold.py 2>&1 | grep -v "Making TMA" | tail -40
stage dense2
timeout 400 python3 -u bench_dense.py --out dense2.jsonl 2>&1 | grep -v "Making TMA" | grep -E "skinny|GPU|done|FAILED" | tail -80
stage tests_gemm
(cd /work/src && timeout 700 python3 -m pytest -q --durations=5 tests/test_sm120_gemm.py -k "not test_filter" 2>&1 | grep -v "Making TMA" | tail -25)
stage tests_regressions_rest
(cd /work/src && timeout 600 python3 -m pytest -q --durations=5 tests/test_sm120_gemm_regressions.py -k "not test_filter and not (m_grouped or k_grouped or sf_branches or swap_boundary or subtile)" 2>&1 | grep -v "Making TMA" | tail -25)
stage end
