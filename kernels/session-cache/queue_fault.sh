L=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
K=/home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/session-cache
cd $K
# Stage 6 (fault-injection calibration of hot_vs_cold.py) in its own GPU-lock hold.
flock $L sg docker -c "FAULT=1 bash $K/run_gpu.sh stage6.sh" > results/logs/stage6.out 2>&1
echo FAULTDONE
