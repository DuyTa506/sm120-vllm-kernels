#!/bin/bash
# Verifier: one GPU-lock hold. Start one server, run the selected verification steps, save logs, remove container.
#   flock $LOCK sg docker -c "kernels/cache-granularity/verify_run.sh <label> <production|optimized> [vllm args]"
# env STEPS: subset of: correct long1 long8 pressure c32   (default: correct long1 long8)
set -u
H=$(cd "$(dirname "$0")" && pwd); D=$(cd $H/../.. && pwd)
LABEL=$1; CFG=$2; shift 2
if [ -z "${PORT:-}" ]; then for p in $(seq 8121 8130); do ss -ltn | grep -q "127.0.0.1:$p " || { PORT=$p; break; }; done; fi
export PORT NAME=${NAME:-cache-granularity-verify}
export BENCH_URL=http://127.0.0.1:$PORT
O=$H/results/verify_$LABEL; mkdir -p $O
STEPS=${STEPS:-"correct long1 long8"}
T0=$(date +%s)
trap 'docker logs $NAME > $O/server_$(date +%H%M%S).log 2>&1; docker rm -f $NAME >/dev/null 2>&1' EXIT
echo "== verify $LABEL CONFIG=$CFG args: $* steps: $STEPS port $PORT ($(date))"
CONFIG=$CFG $D/scripts/serve.sh --enable-prompt-tokens-details "$@" > $O/serve.out 2>&1 || { tail -30 $O/serve.out; exit 1; }
echo "server up after $(( $(date +%s) - T0 ))s"
docker logs $NAME 2>&1 | grep -E "Setting attention block|GPU KV cache size|group sizes|prefix_match" | cut -c1-300 | tee $O/kvinfo.txt
R="python3 $H/replay_subset.py"
LS=$D/results/agentic/trajectories_long_production.json
cd $H
for s in $STEPS; do
  echo "-- $s ($(( $(date +%s) - T0 ))s)"
  case $s in
    correct) timeout 900 python3 $H/verify_correctness.py $O/correct.json $LABEL ;;
    long1)   timeout 600 $R $LS $O/long_c1.json --sessions 0,1,4,7 --conc 1 --tag v$LABEL ;;
    long8)   timeout 600 $R $LS $O/long_c8.json --sessions 0,1,4,7 --conc 8 --tag v$LABEL ;;
    pressure) timeout 900 python3 $H/verify_pressure.py $O/pressure.json $LABEL ;;
    pressrev) timeout 900 python3 $H/verify_pressure.py $O/pressure_rev.json $LABEL --n 56 --order rev ;;
    c32)     timeout 840 python3 $H/verify_c32.py $O/c32_long.json $LABEL ;;
  esac
  echo "   kv usage now: $(curl -s $BENCH_URL/metrics | grep -E '^vllm:kv_cache_usage_perc' | awk '{print $2}')"
done
echo "server errors: $(docker logs $NAME 2>&1 | grep -cE 'Traceback|ERROR|AssertionError')"
echo "done after $(( $(date +%s) - T0 ))s"
