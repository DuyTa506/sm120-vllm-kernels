#!/bin/bash
# A/B decode benchmark in ONE GPU-lock hold: off,on,off,on servers, each: c=16,32,64 x2 (n=12c prompts, 128 in / 512 out).
# results -> results/ab.txt
set -e
D=$(cd "$(dirname "$0")/../.." && pwd); K=$D/kernels/gdn-replay
LOCK=${LOCK:-/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock}
NAME=gdn-replay-srv; PORT=${PORT:-8121}
PKG=/usr/local/lib/python3.12/dist-packages/vllm; G=model_executor/layers/mamba/gdn; B=$K/patch/b/vllm
MOUNTS="-v $B/envs.py:$PKG/envs.py:ro -v $B/$G/qwen_gdn_linear_attn.py:$PKG/$G/qwen_gdn_linear_attn.py:ro -v $B/$G/gdn_state_replay.py:$PKG/$G/gdn_state_replay.py:ro -v $B/$G/gdn_replay.cu:$PKG/$G/gdn_replay.cu:ro -v $K/build/libgdn_replay.so:$PKG/$G/libgdn_replay.so:ro -v $B/v1/worker/gpu/model_states/mamba_hybrid.py:$PKG/v1/worker/gpu/model_states/mamba_hybrid.py:ro"
OUT=$K/results/ab
cat > $OUT.inner.sh <<EOF
M=unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
acc() { curl -s http://127.0.0.1:$PORT/metrics | grep -E "^vllm:spec_decode_num_\$1_tokens_total" | awk '{s+=\$NF} END{print s+0}'; }
for mode in off on off on; do
  ENVF=""; [ \$mode = on ] && ENVF="-e VLLM_GDN_STATE_REPLAY=1"
  FLAGS="$MOUNTS \$ENVF" CONFIG=tuned NAME=$NAME PORT=$PORT $D/scripts/serve.sh >/dev/null 2>&1
  docker exec $NAME vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model \$M --served-model-name bench --dataset-name random --ignore-eos --num-warmups 4 --num-prompts 32 --max-concurrency 16 --random-input-len 128 --random-output-len 256 >/dev/null 2>&1
  a0=\$(acc accepted); d0=\$(acc draft)
  for rep in 1 2; do for c in 16 32 64; do
    r=\$(docker exec $NAME vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model \$M --served-model-name bench --dataset-name random --ignore-eos --num-warmups 1 --num-prompts \$((c*12)) --max-concurrency \$c --random-input-len 128 --random-output-len 512 --seed \$((900+c)) 2>&1 | grep -E 'Output token throughput|Median ITL' | tr -s ' ' | tr '\n' ' ')
    echo "\$mode rep\$rep c=\$c: \$r"
  done; done
  a1=\$(acc accepted); d1=\$(acc draft)
  python3 -c "print('\$mode acceptance', (\$a1-\$a0)/max(\$d1-\$d0,1))"
  docker rm -f $NAME >/dev/null
done
EOF
flock $LOCK sg docker -c "bash $OUT.inner.sh 2>&1 | tee $OUT.txt"
rm -f $OUT.inner.sh
