#!/bin/bash
# usage: run_gpu.sh <logname> <cmd...>  : takes the shared GPU lock, runs cmd in the container, logs to results/<logname>.log
D=$(cd "$(dirname "$0")" && pwd)
LOG=$1; shift
flock /tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock bash -c "date +%T > $D/results/$LOG.started; NAME=dg-sm120-$LOG $D/dg_docker.sh --gpu '$*' > $D/results/$LOG.log 2>&1; echo exit=\$? >> $D/results/$LOG.log; date +%T > $D/results/$LOG.finished"
