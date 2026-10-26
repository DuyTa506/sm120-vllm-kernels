#!/bin/bash
# Runs verify stages, one GPU-lock hold each.  bash verify_queue.sh [A A2 B C D D2 E F]
K=$(cd "$(dirname "$0")" && pwd); cd $K
L=/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock
mkdir -p verify_results/logs
for st in ${@:-A B C D}; do
  echo "== stage $st waiting for lock $(date)"
  case $st in
    A)  flock $L sg docker -c "DEBUGDUMP=1 VENV='-e VLLM_SESSION_STATE_TTL_S=150' bash $K/verify_run_gpu.sh verify_stage_A.sh --prefix-match-unit 16" ;;
    A2) flock $L sg docker -c "DEBUGDUMP=1 VENV='-e VLLM_SESSION_STATE_TTL_S=150' bash $K/verify_run_gpu.sh verify_stage_A2.sh --prefix-match-unit 16" ;;
    B)  flock $L sg docker -c "bash $K/verify_run_gpu.sh verify_stage_B.sh" ;;
    C)  flock $L sg docker -c "bash $K/verify_run_gpu.sh verify_stage_C.sh --prefix-match-unit 16" ;;
    D)  flock $L sg docker -c "bash $K/verify_run_gpu.sh verify_stage_D.sh" ;;
    D2) flock $L sg docker -c "bash $K/verify_run_gpu.sh verify_stage_D2.sh" ;;
    E)  flock $L sg docker -c "bash $K/verify_run_gpu.sh verify_stage_E.sh" ;;
    F)  flock $L sg docker -c "DEBUGDUMP=1 bash $K/verify_run_gpu.sh verify_stage_F.sh" ;;
  esac > verify_results/logs/stage_$st.out 2>&1
  echo "== stage $st done $(date)"
done
