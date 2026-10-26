#!/bin/bash
# Verifier's GPU hold: start a server (patched unless STOCK=1), run one verify stage script, remove the container.
# usage (inside the GPU lock):  sg docker -c "bash verify_run_gpu.sh <verify_stage.sh> [extra vllm serve args]"
# env: STOCK=1 (no patch), DEBUGDUMP=1 (debug variant files for the direct state check),
#      VENV="-e K=V ..." extra docker env (e.g. TTL / MAX of the session cache), CONFIG (default production)
set -u
K=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$K/../.." && pwd)
STAGE=$1; shift
if [ -z "${PORT:-}" ]; then
  for p in $(seq 8121 8130) $(seq 8117 8120); do
    ss -ltn | awk '{print $4}' | grep -q ":$p\$" || { PORT=$p; break; }
  done
fi
export NAME=session-cache-verify PORT CONFIG=${CONFIG:-production}
export BENCH_URL=http://127.0.0.1:$PORT
LOG=$K/verify_results/logs; mkdir -p $LOG
TS=$(date +%Y%m%d-%H%M%S)
cleanup() { docker logs $NAME > $LOG/server-$TS.log 2>&1; docker rm -f $NAME >/dev/null 2>&1; }
trap cleanup EXIT
FLAGS_X=""
if [ "${STOCK:-0}" != "1" ]; then
  FLAGS_X="$(cat $K/.mounts)"
  if [ "${DEBUGDUMP:-0}" = "1" ]; then
    FLAGS_X="${FLAGS_X//$K\/b\/vllm\/v1\/worker\/gpu\/model_states\/mamba_hybrid.py/$K/debug/mamba_hybrid.py}"
    FLAGS_X="${FLAGS_X//$K\/b\/vllm\/v1\/core\/session_state_cache.py/$K/debug/session_state_cache.py}"
    FLAGS_X="$FLAGS_X -v $K:/w:ro -e VLLM_SESSION_STATE_DEBUG_DIR=/tmp/ssdump"
  fi
  FLAGS_X="$FLAGS_X -e VLLM_SESSION_STATE_CACHE=1 -e VLLM_SESSION_STATE_LOG=${SESSION_LOG:-1} ${VENV:-}"
fi
export FLAGS="$FLAGS_X"
echo "[$TS] start $NAME port $PORT CONFIG=$CONFIG STOCK=${STOCK:-0} DEBUGDUMP=${DEBUGDUMP:-0} VENV=${VENV:-} args: $*"
t0=$(date +%s)
$R/scripts/serve.sh --enable-prompt-tokens-details "$@" || { echo "server failed"; exit 1; }
echo "server up after $(( $(date +%s) - t0 ))s"
cd $K
timeout ${STAGE_TIMEOUT:-1020} bash $STAGE
echo "stage exit $? after $(( $(date +%s) - t0 ))s"
docker logs $NAME 2>&1 | grep -E "Traceback|ERROR|Error" | tail -20
docker logs $NAME 2>&1 | grep -E "Session state cache:" | tail -3
