#!/bin/bash
# correctness + benchmark in one GPU hold (run inside the vllm image, cwd = this dir)
set -e
python3 test_correctness.py 2>&1 | grep -v Warn
python3 bench.py "$@" 2>&1 | grep -v Warn
