L=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
K=/home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/session-cache
cd $K
# Stage 7 (prefix-match-unit 16, reversed A/B order, step counts) in its own GPU-lock hold.
flock $L sg docker -c "bash $K/run_gpu.sh stage7.sh --prefix-match-unit 16" > results/logs/stage7.out 2>&1
echo S7DONE
