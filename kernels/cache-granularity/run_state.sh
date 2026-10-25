#!/bin/bash
# Engine-level GDN state check (state_check.py) for one config, inside ONE hold of the GPU lock:
#   flock $LOCK sg docker -c "kernels/cache-granularity/run_state.sh <label> <prod|opt> [vllm args]"
# prod = CONFIG=production, opt = the optimized config (patches 01+02, env flags) rebuilt here so the GDN file can be
# swapped for the copy with the test-only dump hook (build/<kind>/qwen_gdn_linear_attn.py, make_dump_build.py).
set -u
H=$(cd "$(dirname "$0")" && pwd); D=$(cd $H/../.. && pwd)
LABEL=$1; KIND=$2; shift 2
export PORT=${PORT:-8117} NAME=${NAME:-cache-granularity}
export BENCH_URL=http://127.0.0.1:$PORT
PKG=/usr/local/lib/python3.12/dist-packages/vllm
IMG=vllm/vllm-openai:v0.32.0
O=$H/results/$LABEL; mkdir -p $O/dumps; chmod 777 $O/dumps; rm -f $O/dumps/*
trap 'docker logs $NAME > $O/server.log 2>&1; docker rm -f $NAME >/dev/null 2>&1' EXIT
HOOK="-e VLLM_CG_DUMP_DIR=/dumps -v $O/dumps:/dumps
  -v $H/build/$KIND/qwen_gdn_linear_attn.py:$PKG/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:ro"
if [ $KIND = prod ]; then
  CFG=production; export FLAGS="$HOOK"
else
  CFG=tuned; export FLAGS="$HOOK -e VLLM_SM120_FP8_PREFILL=1 -e VLLM_FI_KERNEL_PAGE64=1 -e VLLM_GDN_INTERNAL_CKPT=1
    -v $D/.cache/flashinfer:/root/.cache/flashinfer
    -v $D/build/vllm/v1/attention/backends/flashinfer.py:$PKG/v1/attention/backends/flashinfer.py:ro"
fi
T0=$(date +%s)
echo "== $LABEL ($KIND) args: $*"
CONFIG=$CFG $D/scripts/serve.sh --enable-prompt-tokens-details "$@" > $O/serve.out 2>&1 || { tail -30 $O/serve.out; exit 1; }
echo "server up after $(( $(date +%s) - T0 ))s"
docker logs $NAME 2>&1 | grep -E "Setting attention block|GPU KV cache size|group sizes" | tee $O/kvinfo.txt
timeout 900 docker run --rm --network host --user $(id -u):$(id -g) --entrypoint python3 -e HOME=/tmp -e BENCH_URL=$BENCH_URL \
  -v $D:$D $IMG $H/state_check.py $O/dumps $O/state_check.json $LABEL
echo "done after $(( $(date +%s) - T0 ))s"
