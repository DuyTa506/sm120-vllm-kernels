#!/bin/bash
# Start a production-config server (container kv-offload, port 8121), run one or more client commands against it,
# save the server log, and always remove the container. Run it inside ONE gpu.lock hold:
#   flock <gpu.lock> sg docker -c "OFFLOAD=96 ./run_cfg.sh <label> 'python3 idle_bench.py ...' ['cmd2' ...]"
# env: OFFLOAD=<GiB>   --kv-offloading-size (empty = no offloading, the baseline)
#      BACKEND=native|lmcache (default native); SIMPLE=1 selects SimpleCPUOffloadConnector (native backend)
#      KVT_FILE=<json file> passed verbatim as --kv-transfer-config (e.g. lmcache_mp.json)
#      EXTRA="..."     extra vllm serve args, XFLAGS="..." extra docker args, LIMIT=<s> client time limit (1080)
set -u
D=$(cd "$(dirname "$0")" && pwd); R=$(cd $D/../.. && pwd)
LABEL=$1; shift
NAME=kv-offload; PORT=8121
mkdir -p $D/logs $D/results
cleanup() { docker logs $NAME > $D/logs/$LABEL.server.log 2>&1; docker rm -f $NAME >/dev/null 2>&1; }
trap cleanup EXIT
FLAGS="-e VLLM_SERVER_DEV_MODE=1 -v $D:/work:ro ${XFLAGS:-}"
[ "${SIMPLE:-0}" = "1" ] && FLAGS="$FLAGS -e VLLM_USE_SIMPLE_KV_OFFLOAD=1"
ARGS="--enable-prompt-tokens-details"
[ -n "${OFFLOAD:-}" ] && ARGS="$ARGS --kv-offloading-size $OFFLOAD --kv-offloading-backend ${BACKEND:-native}"
echo "[$(date +%T)] $LABEL: serve $ARGS ${EXTRA:-} FLAGS=$FLAGS"
T0=$(date +%s)
KVT=(); [ -n "${KVT_FILE:-}" ] && KVT=(--kv-transfer-config "$(cat $KVT_FILE)")
CONFIG=production NAME=$NAME PORT=$PORT FLAGS="$FLAGS" $R/scripts/serve.sh $ARGS ${EXTRA:-} "${KVT[@]}" > $D/logs/$LABEL.serve.log 2>&1
rc=$?
echo "[$(date +%T)] $LABEL: server rc=$rc after $(( $(date +%s) - T0 ))s"
[ $rc -ne 0 ] && { tail -40 $D/logs/$LABEL.serve.log; exit 1; }
cd $D
for cmd in "$@"; do
  echo "[$(date +%T)] $LABEL: $cmd"
  BENCH_URL=http://127.0.0.1:$PORT timeout ${LIMIT:-1080} bash -c "$cmd" 2>&1 | tee -a $D/logs/$LABEL.client.log
done
echo "[$(date +%T)] $LABEL: done after $(( $(date +%s) - T0 ))s"
