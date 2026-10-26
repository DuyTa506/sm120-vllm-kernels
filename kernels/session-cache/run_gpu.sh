#!/bin/bash
# One GPU hold: start a server (patched unless STOCK=1), run the given stage script, remove the container.
# usage (inside the GPU lock):  sg docker -c "kernels/session-cache/run_gpu.sh <stage.sh> [extra vllm serve args]"
# env: STOCK=1 (no patch mounts), SESSION=0 (patch mounted but VLLM_SESSION_STATE_CACHE unset),
#      CONFIG (default production), PORT (default 8117), NAME (default session-cache)
set -u
K=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$K/../.." && pwd)
STAGE=$1; shift
if [ -z "${PORT:-}" ]; then  # first free port in 8117..8130 (other agents share this range)
  for p in $(seq 8117 8130); do
    ss -ltn | awk '{print $4}' | grep -q ":$p\$" || { PORT=$p; break; }
  done
fi
export NAME=${NAME:-session-cache} PORT=${PORT:-8117} CONFIG=${CONFIG:-production}
export BENCH_URL=http://127.0.0.1:$PORT
LOG=$K/results/logs; mkdir -p $LOG
TS=$(date +%Y%m%d-%H%M%S)
cleanup() { docker logs $NAME > $LOG/server-$TS.log 2>&1; docker rm -f $NAME >/dev/null 2>&1; }
trap cleanup EXIT
FLAGS_X=""
if [ "${STOCK:-0}" != "1" ]; then
  FLAGS_X="$(cat $K/.mounts)"
  # FAULT=1: mount the fault-injected worker (snapshot one token off) to calibrate hot_vs_cold.py
  [ "${FAULT:-0}" = "1" ] && FLAGS_X="${FLAGS_X//$K\/b\/vllm\/v1\/worker\/gpu\/model_states\/mamba_hybrid.py/$K/fault/mamba_hybrid.py}"
  # DEBUGDUMP=1: debug variant (debug/): snapshot meta + state dumps to /tmp/ssdump in the container,
  # repo mounted read-only at /w for state_check_analyze.py (used by stage8.sh only)
  if [ "${DEBUGDUMP:-0}" = "1" ]; then
    FLAGS_X="${FLAGS_X//$K\/b\/vllm\/v1\/worker\/gpu\/model_states\/mamba_hybrid.py/$K/debug/mamba_hybrid.py}"
    FLAGS_X="${FLAGS_X//$K\/b\/vllm\/v1\/core\/session_state_cache.py/$K/debug/session_state_cache.py}"
    FLAGS_X="$FLAGS_X -v $K:/w:ro -e VLLM_SESSION_STATE_DEBUG_DIR=/tmp/ssdump"
  fi
  [ "${SESSION:-1}" = "1" ] && FLAGS_X="$FLAGS_X -e VLLM_SESSION_STATE_CACHE=1 -e VLLM_SESSION_STATE_LOG=${SESSION_LOG:-50}"
fi
export FLAGS="${FLAGS:-} $FLAGS_X"
echo "[$TS] start $NAME CONFIG=$CONFIG STOCK=${STOCK:-0} SESSION=${SESSION:-1} args: $*"
t0=$(date +%s)
$R/scripts/serve.sh --enable-prompt-tokens-details "$@" || { echo "server failed"; exit 1; }
echo "server up after $(( $(date +%s) - t0 ))s"
cd $K
timeout ${STAGE_TIMEOUT:-1080} bash $STAGE
echo "stage exit $? after $(( $(date +%s) - t0 ))s"
docker logs $NAME 2>&1 | grep -E "Session state cache|session state|Traceback|Error|error" | tail -20
