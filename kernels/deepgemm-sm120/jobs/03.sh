cd /work
export PYTHONPATH=/work/src:/work/src/tests
export DG_PRINT_CONFIGS=1
stage() { echo; echo "=== STAGE $1 start $(date +%T)"; }
stage ours_cold
timeout 700 python3 -u bench_ours_cold.py 2>&1 | grep -v "Making TMA" | tail -40
stage graph_safety_more
timeout 400 python3 -u graph_safety.py --replays 1000 --out graph_safety2.jsonl --modes fp8_labels128_G288_w2,fp8_masked_G288,fp8_masked_G288_w2,fp8_psum128_G288,fp8_masked_w2,fp8_labels128_w2 2>&1 | grep -v "Making TMA" | tail -30
stage grouped_repeat
timeout 300 python3 -u bench_grouped.py --mode prefill --routings L19 --aligns 64,128 --no-fp4 --out grouped_rep.jsonl 2>&1 | grep -E "^   \{|FAILED|Error|Traceback" | head -5
stage tests_gemm
(cd /work/src && timeout 1500 python3 -m pytest -v --durations=5 tests/test_sm120_gemm.py -k "not test_filter" 2>&1 | grep -v "Making TMA" | grep -E "PASSED|FAILED|ERROR|passed|failed|slowest|s call" | sed 's/ \[.*//' | tail -90)
stage end
