set -e
FLAGS='-v /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/gdn-replay/patch/b/vllm/envs.py:/usr/local/lib/python3.12/dist-packages/vllm/envs.py:ro
 -v /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/gdn-replay/patch/b/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:ro
 -v /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/gdn-replay/patch/b/vllm/model_executor/layers/mamba/gdn/gdn_state_replay.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn/gdn_state_replay.py:ro
 -v /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/gdn-replay/patch/b/vllm/model_executor/layers/mamba/gdn/gdn_replay.cu:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn/gdn_replay.cu:ro
 -v /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/gdn-replay/build/libgdn_replay.so:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mamba/gdn/libgdn_replay.so:ro
 -v /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/gdn-replay/patch/b/vllm/v1/worker/gpu/model_states/mamba_hybrid.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_states/mamba_hybrid.py:ro -e VLLM_GDN_STATE_REPLAY=1 -e VLLM_GDN_STATE_REPLAY_CHECK=1' CONFIG=tuned NAME=gdn-replay-srv PORT=8121 /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/scripts/serve.sh -cc.cudagraph_mode=PIECEWISE
export BENCH_URL=http://127.0.0.1:8121
echo "== server log (replay lines)"; docker logs gdn-replay-srv 2>&1 | grep -iE "replay|GDN spec decode kernel|Mamba cache mode|attention block size|Compiling model again|Directly load AOT" | cut -c1-220 || true
curl -s $BENCH_URL/v1/completions -H 'Content-Type: application/json' -d '{"model":"bench","prompt":"hello","max_tokens":8,"temperature":0}' >/dev/null
echo "== probe (cold, 2 passes + long generation crossing the 2176 block boundary)"
python3 -I /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/gdn-replay/tests/probe.py /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/gdn-replay/results/e2e_check_v2.probeA.json checkA_v2 long
python3 -I /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/gdn-replay/tests/probe.py /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/gdn-replay/results/e2e_check_v2.probeB.json checkB_v2
python3 -I /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/gdn-replay/tests/compare_probe.py /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/gdn-replay/results/e2e_check_v2.probeA.json /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/gdn-replay/results/e2e_check_v2.probeB.json | tail -3
a0=$(curl -s $BENCH_URL/metrics | grep -E '^vllm:spec_decode_num_accepted_tokens_total' | awk '{s+=$NF} END{print s+0}' || true)
d0=$(curl -s $BENCH_URL/metrics | grep -E '^vllm:spec_decode_num_draft_tokens_total' | awk '{s+=$NF} END{print s+0}')
if [ "check" = check ]; then
  echo "== short bench (check mode is slow: c=1 and c=8 only)"
  for c in 1 8; do
    docker exec gdn-replay-srv vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --served-model-name bench --dataset-name random --ignore-eos --num-warmups 1 --num-prompts $((c*3)) --max-concurrency $c --random-input-len 128 --random-output-len 160 --seed 77 2>&1 | grep -E 'Output token throughput|Median ITL' | tr -s ' ' | tr '\n' ' '; echo
  done
else
  echo "== bench_decode (c=1,4,16,64)"
  /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/bench/bench_decode.sh gdn-replay-srv
fi
a1=$(curl -s $BENCH_URL/metrics | grep -E '^vllm:spec_decode_num_accepted_tokens_total' | awk '{s+=$NF} END{print s+0}')
d1=$(curl -s $BENCH_URL/metrics | grep -E '^vllm:spec_decode_num_draft_tokens_total' | awk '{s+=$NF} END{print s+0}')
python3 - <<PY
a0,a1,d0,d1=$a0,$a1,$d0,$d1
print(f"spec decode over bench_decode: accepted {a1-a0:.0f} / drafted {d1-d0:.0f} = {(a1-a0)/max(d1-d0,1):.4f}")
PY
echo "== stress (finish mid-step, aborts)"
python3 -I /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/gdn-replay/tests/stress.py 60 8 3000
echo "preemptions: $(curl -s $BENCH_URL/metrics | grep -E '^vllm:num_preemptions_total' | awk '{s+=$NF} END{print s+0}')"
if [ "check" = on ]; then
  echo "== prefix-cache checkpoint check (eval/ckpt_dist.py)"
  python3 -I /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/eval/ckpt_dist.py gdnreplay_v2
fi
echo "== post-stress probe"
python3 -I /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/gdn-replay/tests/probe.py /home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/gdn-replay/results/e2e_check_v2.probeC.json checkC_v2
echo "== errors / check lines"
docker logs gdn-replay-srv 2>&1 | grep -iE "Traceback|Error|CHECK|replay check|mismatch" | sort | uniq -c | sort -rn | head -12 || true
