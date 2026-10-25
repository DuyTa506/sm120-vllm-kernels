#!/bin/bash
# Start one server config, run the granularity measurements against it, remove the container.
# Run inside ONE hold of the GPU lock:
#   flock $LOCK sg docker -c "kernels/cache-granularity/run_config.sh <label> <production|optimized> [vllm args]"
# env: STEPS   space-separated subset of: short1 short8 long1 long8 hotcold ckpt  (default: all)
#      PORT    (8117)   NAME (cache-granularity)
set -u
H=$(cd "$(dirname "$0")" && pwd); D=$(cd $H/../.. && pwd)
LABEL=$1; CFG=$2; shift 2
export PORT=${PORT:-8117} NAME=${NAME:-cache-granularity}
export BENCH_URL=http://127.0.0.1:$PORT
O=$H/results/$LABEL; mkdir -p $O
STEPS=${STEPS:-"short1 short8 long1 long8 hotcold"}
T0=$(date +%s)
trap 'docker logs $NAME > $O/server.log 2>&1; docker rm -f $NAME >/dev/null 2>&1' EXIT
echo "== $LABEL CONFIG=$CFG args: $* steps: $STEPS"
CONFIG=$CFG $D/scripts/serve.sh --enable-prompt-tokens-details "$@" > $O/serve.out 2>&1 || { cat $O/serve.out; exit 1; }
echo "server up after $(( $(date +%s) - T0 ))s"
docker logs $NAME 2>&1 | grep -E "block size|block_size|prefix|hash|GPU KV cache size|Maximum concurrency|Padding mamba|kv cache group|lcm" | grep -v "Uvicorn\|Route" | tail -20 | tee $O/kvinfo.txt
R="python3 $H/replay_subset.py"
LS=$D/results/agentic/trajectories_long_production.json
SS=$D/results/agentic/trajectories_short_production.json
for s in $STEPS; do
  echo "-- $s ($(( $(date +%s) - T0 ))s)"
  case $s in
    short1) timeout 600 $R $SS $O/short_c1.json --conc 1 --tag $LABEL ;;
    short8) timeout 600 $R $SS $O/short_c8.json --conc 8 --tag $LABEL ;;
    long1)  timeout 600 $R $LS $O/long_c1.json --sessions 0,1,4,7 --conc 1 --tag $LABEL ;;
    long8)  timeout 600 $R $LS $O/long_c8.json --sessions 0,1,4,7 --conc 8 --tag $LABEL ;;
    hotcold) timeout 600 python3 $H/hot_cold.py $O/hotcold.json $LABEL ;;
    ckpt)   timeout 300 python3 $D/eval/ckpt_dist.py $LABEL | tee $O/ckpt_dist.txt
            [ -f $O/hotcold.json ] || timeout 600 python3 $H/hot_cold.py $O/hotcold.json $LABEL ;;
  esac
done
echo "done after $(( $(date +%s) - T0 ))s"
