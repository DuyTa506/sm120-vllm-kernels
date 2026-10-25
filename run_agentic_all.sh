#!/bin/bash
# Agentic benchmark for every config. Trajectories are recorded live on the first config and replayed
# token-for-token on all of them; every config also records its own live run (tool-call validity, completion).
# Two task sets: short coding sessions (10-30k tokens) and long whole-package reviews (up to 131k tokens).
# Each config runs inside one hold of the GPU lock (LOCK=path, optional) so it can interleave with other users.
# usage: ./run_agentic_all.sh [configs...]     env: CONCS (short replay), LONG_CONCS (long replay)
D=$(cd "$(dirname "$0")" && pwd)
O=$D/results/agentic; mkdir -p $O
TOOLFLAGS="--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3"
CONFIGS=${@:-production tuned optimized}
FIRST=$(echo $CONFIGS | cut -d' ' -f1)
B="python3 -I $D/bench/agentic_bench.py"
for c in $CONFIGS; do
  echo "=== $c"
  ${LOCK:+flock $LOCK} bash -c "
    CONFIG=$c $D/scripts/serve.sh $TOOLFLAGS || exit 1
    docker logs bench-vllm 2>&1 | grep -E 'GPU KV cache size|Maximum concurrency' | tail -2
    for kind in short long; do
      $B record $O/trajectories_\${kind}_$c.json \$kind || { docker rm -f bench-vllm; exit 1; }
      [ \$kind = short ] && concs=${CONCS:-1,4,8,16,32} || concs=${LONG_CONCS:-1,8,32}
      $B replay $O/trajectories_\${kind}_$FIRST.json $O/replay_\${kind}_$c.json \$concs
    done
    docker rm -f bench-vllm >/dev/null
  " || exit 1
done
