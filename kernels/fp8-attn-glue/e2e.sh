#!/bin/bash
# End-to-end check of patch 03 in the real server ("optimized" config of scripts/serve.sh,
# MTP-2, prefix caching off for the timed prompts by using fresh random seeds).
# For VLLM_SM120_FP8_PREFILL_NOGLUE=0 and =1: start server, greedy 8k-token completion
# (tokens + top-5 logprobs saved), 8k prefill TTFT (vllm bench serve, 8 prompts, c=1).
# Runs entirely under the GPU lock; leaves no container behind.
set -u
LOCK=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
R=/home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels
H=$R/kernels/fp8-attn-glue
OUT=${OUT:-$H/results}
mkdir -p $OUT $H/.cache/vllm $H/.cache/flashinfer
PKG=/usr/local/lib/python3.12/dist-packages/vllm
NAME=fp8-attn-glue-serve
PORT=8157
exec flock $LOCK sg docker -c "
set -u
for CFG in 0:0 1:0 1:1; do
  NG=\${CFG%%:*}; VF=\${CFG##*:}
  docker rm -f $NAME >/dev/null 2>&1
  docker run -d --name $NAME --gpus device=0 --ipc=host -p 127.0.0.1:$PORT:8000 \
    -e HF_HUB_OFFLINE=1 -e VLLM_SM120_FP8_PREFILL=1 -e VLLM_FI_KERNEL_PAGE64=1 -e VLLM_GDN_INTERNAL_CKPT=1 \
    -e VLLM_SM120_FP8_PREFILL_NOGLUE=\$NG -e VLLM_SM120_FP8_PREFILL_NOGLUE_VERIFY=\$VF \
    -v /opt/intramind/hf-cache:/root/.cache/huggingface:ro \
    -v $H/.cache/vllm:/root/.cache/vllm -v $H/.cache/flashinfer:/root/.cache/flashinfer \
    -v $H/patch/b/vllm/v1/attention/backends/flashinfer.py:$PKG/v1/attention/backends/flashinfer.py:ro \
    -v $H/patch/b/vllm/v1/attention/backends/sm120_fp8_fmha_glue.py:$PKG/v1/attention/backends/sm120_fp8_fmha_glue.py:ro \
    -v $R/build/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:$PKG/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:ro \
    -v $H:/glue:ro \
    --entrypoint vllm vllm/vllm-openai:v0.32.0 serve unsloth/Qwen3.6-35B-A3B-NVFP4-Fast \
    --served-model-name bench --language-model-only --max-model-len 131072 --max-num-seqs 64 \
    --gpu-memory-utilization 0.78 --enable-chunked-prefill --enable-prefix-caching \
    --speculative-config '{\"method\":\"mtp\",\"num_speculative_tokens\":2}' --async-scheduling \
    --max-num-batched-tokens 16384 --cudagraph-capture-sizes \$(cat $R/scripts/cudagraph_sizes.txt) >/dev/null
  ok=0
  for i in \$(seq 1 120); do
    curl -sf http://127.0.0.1:$PORT/health >/dev/null && { ok=1; break; }
    docker ps -q -f name=^$NAME\$ | grep -q . || break
    sleep 5
  done
  if [ \$ok = 1 ] && [ \$VF = 1 ]; then
    docker exec $NAME python3 /glue/e2e_client.py --port 8000 --varlen > /dev/null
    docker exec $NAME python3 /glue/e2e_client.py --port 8000 --out /dev/null
    docker logs $NAME 2>&1 | grep NOGLUE_VERIFY > $OUT/e2e_verify.txt
  elif [ \$ok = 1 ]; then
    docker exec $NAME python3 /glue/e2e_client.py --port 8000 --out /dev/stdout > $OUT/e2e_greedy_noglue\$NG.json
    docker exec $NAME vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 \
      --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --served-model-name bench --dataset-name random \
      --ignore-eos --num-warmups 2 --num-prompts 16 --seed 4242 --max-concurrency 1 \
      --random-input-len 8192 --random-output-len 1 2>&1 | grep -E 'TTFT' > $OUT/e2e_ttft8k_noglue\$NG.txt
    docker logs $NAME 2>&1 | grep -iE 'error|traceback' | head -20 > $OUT/e2e_errors_noglue\$NG.txt
  else
    docker logs --tail 60 $NAME > $OUT/e2e_failed_\$NG\$VF.txt 2>&1
  fi
  docker rm -f $NAME >/dev/null 2>&1
done
"
