L=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
K=/home/aiqk3/Desktop/duy_dev/training_space/sm120-vllm-kernels/kernels/session-cache
cd $K
# usage: bash queue_all.sh [stages...]   (default: 2 3 4 5); one GPU-lock hold per stage
for s in ${@:-2 3 4 5}; do
  case $s in
    4) flock $L sg docker -c "bash $K/run_gpu.sh stage4.sh --prefix-match-unit 16" > results/logs/stage4.out 2>&1 ;;
    5) flock $L sg docker -c "STOCK=1 bash $K/run_gpu.sh stage5.sh" > results/logs/stage5.out 2>&1 ;;
    *) flock $L sg docker -c "bash $K/run_gpu.sh stage$s.sh" > results/logs/stage$s.out 2>&1 ;;
  esac
done
echo ALLDONE
