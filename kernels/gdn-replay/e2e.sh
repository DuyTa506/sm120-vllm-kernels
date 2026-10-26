#!/bin/bash
# End-to-end run of one mode in ONE GPU-lock hold (server start ~3-5 min + benches).
# usage: e2e.sh <off|on|check> [tag]      CONFIG=tuned server (stock tree + this patch mounted; flag decides)
#   off    VLLM_GDN_STATE_REPLAY unset  (patched files mounted, flag default)
#   on     VLLM_GDN_STATE_REPLAY=1
#   check  VLLM_GDN_STATE_REPLAY=1 VLLM_GDN_STATE_REPLAY_CHECK=1, PIECEWISE graphs: every GDN layer call
#          runs replay and the stock kernel and compares outputs and states bitwise (slow, debug)
# results -> results/e2e_<mode><tag>.*
set -e
D=$(cd "$(dirname "$0")/../.." && pwd)
K=$D/kernels/gdn-replay
MODE=${1:-off}; TAG=${2:-}
LOCK=${LOCK:-/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock}
NAME=gdn-replay-srv; PORT=${PORT:-8121}
PKG=/usr/local/lib/python3.12/dist-packages/vllm
G=model_executor/layers/mamba/gdn
B=$K/patch/b/vllm
OUT=$K/results/e2e_$MODE$TAG
mkdir -p $K/results $D/.cache/vllm
MOUNTS="-v $B/envs.py:$PKG/envs.py:ro
 -v $B/$G/qwen_gdn_linear_attn.py:$PKG/$G/qwen_gdn_linear_attn.py:ro
 -v $B/$G/gdn_state_replay.py:$PKG/$G/gdn_state_replay.py:ro
 -v $B/$G/gdn_replay.cu:$PKG/$G/gdn_replay.cu:ro
 -v $K/build/libgdn_replay.so:$PKG/$G/libgdn_replay.so:ro
 -v $B/v1/worker/gpu/model_states/mamba_hybrid.py:$PKG/v1/worker/gpu/model_states/mamba_hybrid.py:ro"
ENVF=""; EXTRA=""
case $MODE in
  on) ENVF="-e VLLM_GDN_STATE_REPLAY=1" ;;
  check) ENVF="-e VLLM_GDN_STATE_REPLAY=1 -e VLLM_GDN_STATE_REPLAY_CHECK=1"; EXTRA="-cc.cudagraph_mode=PIECEWISE" ;;
  onpw) ENVF="-e VLLM_GDN_STATE_REPLAY=1"; EXTRA="-cc.cudagraph_mode=PIECEWISE" ;;
  shadow) ENVF="-e VLLM_GDN_STATE_REPLAY=1 -e VLLM_GDN_STATE_REPLAY_CHECK=2 -e VLLM_GDN_STATE_REPLAY_MIN_REQS=2"; EXTRA="-cc.cudagraph_mode=PIECEWISE" ;;
  off) ;;
  *) echo "mode?"; exit 1 ;;
esac
[ -n "$KVBLOCKS" ] && EXTRA="$EXTRA --num-gpu-blocks-override $KVBLOCKS"
metric() { curl -s http://127.0.0.1:$PORT/metrics | grep -E "^$1(\{|\s)" | awk '{s+=$NF} END{print s+0}'; }
cat > $OUT.inner.sh <<EOF
set -e
FLAGS='$MOUNTS $ENVF' CONFIG=tuned NAME=$NAME PORT=$PORT $D/scripts/serve.sh $EXTRA
export BENCH_URL=http://127.0.0.1:$PORT
echo "== server log (replay lines)"; docker logs $NAME 2>&1 | grep -iE "replay|GDN spec decode kernel|Mamba cache mode|attention block size|Compiling model again|Directly load AOT" | cut -c1-220 || true
curl -s \$BENCH_URL/v1/completions -H 'Content-Type: application/json' -d '{"model":"bench","prompt":"hello","max_tokens":8,"temperature":0}' >/dev/null
echo "== probe (cold, 2 passes + long generation crossing the 2176 block boundary)"
python3 -I $K/tests/probe.py $OUT.probeA.json ${MODE}A$TAG long
python3 -I $K/tests/probe.py $OUT.probeB.json ${MODE}B$TAG
python3 -I $K/tests/compare_probe.py $OUT.probeA.json $OUT.probeB.json | tail -3
a0=\$(curl -s \$BENCH_URL/metrics | grep -E '^vllm:spec_decode_num_accepted_tokens_total' | awk '{s+=\$NF} END{print s+0}' || true)
d0=\$(curl -s \$BENCH_URL/metrics | grep -E '^vllm:spec_decode_num_draft_tokens_total' | awk '{s+=\$NF} END{print s+0}')
if [ "$MODE" = check ] || [ "$MODE" = shadow ] || [ "$MODE" = onpw ]; then
  echo "== short bench (check mode is slow: c=1 and c=8 only)"
  for c in 1 16; do
    docker exec $NAME vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --served-model-name bench --dataset-name random --ignore-eos --num-warmups 1 --num-prompts \$((c*3)) --max-concurrency \$c --random-input-len 128 --random-output-len 160 --seed 77 2>&1 | grep -E 'Output token throughput|Median ITL' | tr -s ' ' | tr '\n' ' '; echo
  done
else
  echo "== bench_decode (c=1,4,16,64)"
  $D/bench/bench_decode.sh $NAME
fi
a1=\$(curl -s \$BENCH_URL/metrics | grep -E '^vllm:spec_decode_num_accepted_tokens_total' | awk '{s+=\$NF} END{print s+0}')
d1=\$(curl -s \$BENCH_URL/metrics | grep -E '^vllm:spec_decode_num_draft_tokens_total' | awk '{s+=\$NF} END{print s+0}')
python3 - <<PY
a0,a1,d0,d1=\$a0,\$a1,\$d0,\$d1
print(f"spec decode over bench_decode: accepted {a1-a0:.0f} / drafted {d1-d0:.0f} = {(a1-a0)/max(d1-d0,1):.4f}")
PY
echo "== stress (finish mid-step, aborts)"
python3 -I $K/tests/stress.py ${STRESS_N:-200} ${STRESS_C:-24} ${STRESS_P:-3000}
echo "preemptions: \$(curl -s \$BENCH_URL/metrics | grep -E '^vllm:num_preemptions_total' | awk '{s+=\$NF} END{print s+0}')"
if [ "$MODE" = on ] || [ "$MODE" = onpw ]; then
  echo "== prefix-cache checkpoint check (eval/ckpt_dist.py)"
  python3 -I $D/eval/ckpt_dist.py gdnreplay$TAG
fi
echo "== post-stress probe"
python3 -I $K/tests/probe.py $OUT.probeC.json ${MODE}C$TAG
echo "== errors / check lines"
docker logs $NAME 2>&1 | grep -iE "Traceback|Error|CHECK|replay check|mismatch|SHADOW" | sort | uniq -c | sort -rn | head -12 || true
EOF
flock $LOCK sg docker -c "bash $OUT.inner.sh 2>&1 | tee $OUT.txt; docker logs $NAME > $OUT.server.log 2>&1; docker rm -f $NAME >/dev/null"
rm -f $OUT.inner.sh
