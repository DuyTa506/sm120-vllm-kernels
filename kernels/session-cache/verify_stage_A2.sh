# Verify stage A2 (rerun of A without the leak part; analyzer crashed in run 1) (DEBUGDUMP=1, --prefix-match-unit 16, VENV TTL=200): direct state check under concurrency,
# forks, abort+retry, history divergence; then block-leak check after TTL.
# run: flock $L sg docker -c "DEBUGDUMP=1 VENV='-e VLLM_SESSION_STATE_TTL_S=150' bash verify_run_gpu.sh verify_stage_A.sh --prefix-match-unit 16"
set -x
S=../../results/agentic/trajectories_short_production.json
O=verify_results/A; mkdir -p $O
python3 verify_state_conc.py $S $O/cases.json --sessions 0,1,2,3,4,5 --turns 7
docker exec $NAME python3 /w/state_check_analyze.py /tmp/ssdump /w/$O/cases.json > $O/state_check.json
python3 -c "import json; print(json.dumps(json.load(open('$O/state_check.json'))['summary'])[:3000])"
docker logs $NAME 2>&1 | grep "Session state cache:" | tail -2
