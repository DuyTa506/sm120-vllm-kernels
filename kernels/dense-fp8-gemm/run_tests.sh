#!/bin/bash
# In-container test driver (see README): kernel correctness vs vLLM + smoke test of the vLLM patch op.
set -x
python3 /work/kernels/dense-fp8-gemm/correctness.py --impl skinny
VLLM_SM120_SKINNY_FP8=1 VLLM_CACHE_ROOT=/work/kernels/dense-fp8-gemm/.build/vllm-cache \
  python3 /work/kernels/dense-fp8-gemm/test_patch_op.py
