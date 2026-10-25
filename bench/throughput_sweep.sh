#!/bin/bash
# Serving throughput over several workload shapes (vllm bench serve, random dataset, fixed seeds so
# every config sees the same prompts). Writes one TSV row per scenario.
# usage: bench/throughput_sweep.sh <out.tsv> [container]
OUT=$1; C=${2:-bench-vllm}
M=unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
b() { docker exec $C vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model $M \
        --served-model-name bench --dataset-name random --ignore-eos --num-warmups 0 "$@" 2>&1; }
get() { grep -m1 "$1" <<<"$2" | awk -F: '{gsub(/ /,"",$2); print $2}'; }

#        name           in     out   conc  n    extra
SCEN=(
  "long-doc       32768  128   2     8    "
  "prefill-heavy  8192   64    8     48   "
  "rag            4096   256   16    96   "
  "shared-prefix  1024   256   16    96   --random-prefix-len 4096"
  "mixed-len      4096   512   32    128  --random-range-ratio 0.9"
  "chat           1024   512   32    128  "
  "decode-heavy   128    1024  64    192  "
)
printf "scenario\tin\tout\tconc\treq_s\ttotal_tok_s\tout_tok_s\tmed_ttft_ms\tp99_ttft_ms\tmed_tpot_ms\n" > $OUT
for i in "${!SCEN[@]}"; do
  read -r name in out conc n extra <<<"${SCEN[$i]}"
  # warm-up with a different seed: same shapes, no prefix-cache reuse of the measured prompts
  b --random-input-len $in --random-output-len $out --max-concurrency $conc --num-prompts $conc \
    --seed $((9000 + i)) $extra >/dev/null
  r=$(b --random-input-len $in --random-output-len $out --max-concurrency $conc --num-prompts $n \
      --seed $((100 + i)) $extra)
  row="$name\t$in\t$out\t$conc\t$(get 'Request throughput' "$r")\t$(get 'Total token throughput' "$r")"
  row="$row\t$(get 'Output token throughput' "$r")\t$(get 'Median TTFT' "$r")\t$(get 'P99 TTFT' "$r")\t$(get 'Median TPOT' "$r")"
  printf "$row\n" | tee -a $OUT
done
