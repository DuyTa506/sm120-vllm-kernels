set -e
T0=$(date +%s)
FLAGS='-e VLLM_SKINNY_BF16_GEMM=0 -v /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/decode-roofline/vllm/model_executor/layers/utils.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/utils.py:ro -v /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/decode-roofline/vllm/model_executor/layers/skinny_bf16_gemm.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/skinny_bf16_gemm.py:ro' CONFIG=optimized NAME=decode-roofline-verify PORT=8106 /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/decode-roofline/serve_nsys.sh
echo "server up after $(($(date +%s)-T0)) s"
B="docker exec decode-roofline-verify vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --served-model-name bench --dataset-name random --ignore-eos --random-input-len 128 --random-output-len 512"
G='Output token throughput|Median TPOT|Median ITL|Acceptance length'
$B --num-warmups 2 --num-prompts 4 --max-concurrency 1 --seed 1 >/dev/null 2>&1
if [ "1" = 1 ]; then
  echo "greedy c=1 seed 501: $($B --temperature 0 --num-warmups 1 --num-prompts 16 --seed 501 --max-concurrency 1 2>&1 | grep -E "$G" | tr -s ' ' | tr '\n' ' ')"
else
  for s in 501 601 701; do
    echo "greedy c=1 seed $s: $($B --temperature 0 --num-warmups 1 --num-prompts 16 --seed $s --max-concurrency 1 2>&1 | grep -E "$G" | tr -s ' ' | tr '\n' ' ')"
  done
  for s in 501 601; do
    echo "sampling c=1 seed $s: $($B --num-warmups 1 --num-prompts 16 --seed $s --max-concurrency 1 2>&1 | grep -E "$G" | tr -s ' ' | tr '\n' ' ')"
  done
  echo "greedy c=2: $($B --temperature 0 --num-warmups 2 --num-prompts 12 --seed 502 --max-concurrency 2 2>&1 | grep -E "$G" | tr -s ' ' | tr '\n' ' ')"
  echo "greedy c=4: $($B --temperature 0 --num-warmups 2 --num-prompts 24 --seed 504 --max-concurrency 4 2>&1 | grep -E "$G" | tr -s ' ' | tr '\n' ' ')"
fi
echo "skinny triton kernel compiled in this container: $(docker exec decode-roofline-verify bash -c 'grep -rl _skinny_bf16_gemm_kernel /root/.triton /root/.cache 2>/dev/null | grep -v huggingface | head -1' || true)"
echo "total hold $(($(date +%s)-T0)) s"
