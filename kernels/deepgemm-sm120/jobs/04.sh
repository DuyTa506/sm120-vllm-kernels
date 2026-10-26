cd /work
export PYTHONPATH=/work/src:/work/src/tests
export DG_PRINT_CONFIGS=1
echo "=== fp4 warm"; for i in 0 1 2 3; do python3 -u bench_grouped.py --warm-only --fmts fp4 --slice $i/4 > results/warm4_$i.log 2>&1 & done; wait
echo "=== fp4 timed"; timeout 600 python3 -u bench_grouped.py --fmts fp4 --out grouped_fp4.jsonl 2>&1 | grep -E "FAILED|Traceback|Error" | head
echo done
echo "=== dense extra M (c=16/64 MTP-2 sizes)"; timeout 500 python3 -u bench_dense.py --shapes lm_head,gdn_qkvz,attn_qkv,gdn_out --ms 32,48,96,128,192 --out dense3.jsonl 2>&1 | grep -E "FAILED|Traceback|Error|skinny" | head
echo end
