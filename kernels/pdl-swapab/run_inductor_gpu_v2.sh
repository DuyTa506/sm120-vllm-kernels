#!/bin/bash
# v2: keep the inductor cache in results/ (so the generated code can be grepped for gdc_wait), count overlapping kernels.
D=$(cd "$(dirname "$0")" && pwd)
LOCK=${LOCK:-/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock}
mkdir -p $D/results/ind_cache0 $D/results/ind_cache1 
exec flock "$LOCK" sg docker -c "docker run --rm --name pdlswapab-inductor2 --gpus device=0 --ipc=host -u $(id -u):$(id -g) -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 \
  -e TRITON_CACHE_DIR=/tmp/tri -v $D:/work -w /work --entrypoint bash vllm/vllm-openai:v0.32.0 -c \
  'for r in 0 1; do TORCHINDUCTOR_ENABLE_PDL=\$r TORCHINDUCTOR_CACHE_DIR=/work/results/ind_cache\$r timeout 300 python3 pdl_inductor.py 12 4; done' > $D/results/pdl_inductor_v2.txt 2>&1"
