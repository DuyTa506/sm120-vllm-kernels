# One lock hold: parallel JIT warm-up, dense, grouped, CUDA-graph safety, then DeepGEMM's own SM120 tests.
cd /work
export PYTHONPATH=/work/src:/work/src/tests
export DG_PRINT_CONFIGS=1
stage() { echo; echo "=== STAGE $1 start $(date +%T)"; }
python3 -c "import deep_gemm,torch;print(deep_gemm.__file__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), 'SMs', deep_gemm.get_num_sms())"
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,memory.used --format=csv
stage warm
for i in 0 1 2 3 4 5; do
  python3 -u bench_dense.py --warm-only --slice $i/6 > results/warm_dense_$i.log 2>&1 &
  python3 -u bench_grouped.py --warm-only --slice $i/6 > results/warm_grouped_$i.log 2>&1 &
done
wait
tail -qn 1 results/warm_*.log
stage graph_safety
timeout 900 python3 -u graph_safety.py --replays 1000 2>&1 | tail -80
stage grouped
timeout 1200 python3 -u bench_grouped.py 2>&1 | tail -500
stage dense
timeout 700 python3 -u bench_dense.py 2>&1 | tail -500
stage tests
(cd /work/src && timeout 600 python3 -m pytest -q -x --durations=8 tests/test_sm120_gemm.py 2>&1 | tail -40)
(cd /work/src && timeout 240 python3 -m pytest -q --durations=5 tests/test_sm120_gemm_regressions.py -k "m_grouped or k_grouped or sf_branches or swap_boundary or subtile" 2>&1 | tail -25)
stage end
nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv
