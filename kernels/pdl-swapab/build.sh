#!/bin/bash
cd "$(dirname "$0")/pdl_bench" && nvcc -O3 -std=c++17 -gencode=arch=compute_120a,code=sm_120a pdl_chain.cu -o pdl_chain_v2
