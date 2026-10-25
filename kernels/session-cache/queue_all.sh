L=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
K=/home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/session-cache
cd $K
flock $L sg docker -c "bash $K/run_gpu.sh stage2.sh" > results/logs/stage2.out 2>&1
flock $L sg docker -c "bash $K/run_gpu.sh stage3.sh" > results/logs/stage3.out 2>&1
flock $L sg docker -c "bash $K/run_gpu.sh stage4.sh --prefix-match-unit 16" > results/logs/stage4.out 2>&1
flock $L sg docker -c "STOCK=1 bash $K/run_gpu.sh stage5.sh" > results/logs/stage5.out 2>&1
echo ALLDONE
