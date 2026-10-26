#!/bin/bash
# Verifier's server wrapper (independent copy of run_cfg.sh with its own container/port).
# Starts a production-config server (container kv-offload-verify, port 8123), runs client commands, saves the
# server log, always removes the container. Run inside ONE gpu.lock hold:
#   flock <gpu.lock> sg docker -c "OFFLOAD=96 ./verify_run.sh <label> 'python3 verify_seq.py ...'"
# env: OFFLOAD=<GiB> (empty = no offloading), EXTRA="..." extra vllm args, LIMIT=<s> per client command (900)
set -u
D=$(cd "$(dirname "$0")" && pwd); R=$(cd $D/../.. && pwd)
LABEL=$1; shift
NAME=kv-offload-verify; PORT=8123
mkdir -p $D/verify_logs $D/verify_results
cleanup() { docker logs $NAME > $D/verify_logs/$LABEL.server.log 2>&1; docker rm -f $NAME >/dev/null 2>&1;
            echo "[$(date +%T)] $LABEL: container removed; /dev/shm offload files left:"; ls /dev/shm | grep -c vllm_offload; }
trap cleanup EXIT
FLAGS="-e VLLM_SERVER_DEV_MODE=1"
ARGS="--enable-prompt-tokens-details"
[ -n "${OFFLOAD:-}" ] && ARGS="$ARGS --kv-offloading-size $OFFLOAD --kv-offloading-backend native"
echo "[$(date +%T)] $LABEL: serve $ARGS ${EXTRA:-}"
T0=$(date +%s)
for i in $(seq 1 60); do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$FREE" -ge 80000 ] && break
  [ $i = 1 ] && echo "[$(date +%T)] $LABEL: only $FREE MiB free, waiting"; sleep 10
done
[ "$FREE" -ge 80000 ] || { echo "[$(date +%T)] $LABEL: GPU still busy ($FREE MiB free), giving up"; exit 1; }
CONFIG=production NAME=$NAME PORT=$PORT FLAGS="$FLAGS" $R/scripts/serve.sh $ARGS ${EXTRA:-} > $D/verify_logs/$LABEL.serve.log 2>&1
rc=$?
echo "[$(date +%T)] $LABEL: server rc=$rc after $(( $(date +%s) - T0 ))s"
[ $rc -ne 0 ] && { tail -40 $D/verify_logs/$LABEL.serve.log; exit 1; }
cd $D
for cmd in "$@"; do
  echo "[$(date +%T)] $LABEL: $cmd"
  BENCH_URL=http://127.0.0.1:$PORT timeout ${LIMIT:-900} bash -c "$cmd" 2>&1 | tee -a $D/verify_logs/$LABEL.client.log
  echo "[$(date +%T)] host mem: $(free -g | awk '/Mem/{print "used",$3,"avail",$7}')  container RSS: $(docker stats --no-stream --format '{{.MemUsage}}' $NAME)"
done
echo "[$(date +%T)] $LABEL: done after $(( $(date +%s) - T0 ))s"
