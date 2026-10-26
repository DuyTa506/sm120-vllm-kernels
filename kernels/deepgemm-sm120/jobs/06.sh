cd /work
export PYTHONPATH=/work/src:/work/src/tests
timeout 400 python3 -u bench_ours_kernels.py 2>&1 | grep -E "^T=|call sequence|total kernel" 
for bn in 64 128; do echo "== DG_FORCE_BM32_BN=$bn"; DG_FORCE_BM32_BN=$bn timeout 300 python3 -u bench_dg_kernels.py --decode-only 2>&1 | grep -E "^T=|Error|Traceback|assert" ; done
