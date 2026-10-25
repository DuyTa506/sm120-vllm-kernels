#!/bin/bash
# One GPU hold: start a server (patched unless STOCK=1), run the given stage script, remove the container.
# usage (inside the GPU lock):  sg docker -c "kernels/session-cache/run_gpu.sh <stage.sh> [extra vllm serve args]"
# env: STOCK=1 (no patch mounts), SESSION=0 (patch mounted but VLLM_SESSION_STATE_CACHE unset),
#      CONFIG (default production), PORT (default 8117), NAME (default session-cache)
set -u
K=$(cd "$(dirname "$0")" && pwd)
R=$(cd "$K/../.." && pwd)
STAGE=$1; shift
export NAME=${NAME:-session-cache} PORT=${PORT:-8117} CONFIG=${CONFIG:-production}
export BENCH_URL=http://127.0.0.1:$PORT
LOG=$K/results/logs; mkdir -p $LOG
TS=$(date +%Y%m%d-%H%M%S)
cleanup() { docker logs $NAME > $LOG/server-$TS.log 2>&1; docker rm -f $NAME >/dev/null 2>&1; }
trap cleanup EXIT
FLAGS_X=""
if [ "${STOCK:-0}" != "1" ]; then
  FLAGS_X="$(cat $K/.mounts)"
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
