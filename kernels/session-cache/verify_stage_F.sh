# Verify stage F (DEBUGDUMP=1, production config): direct state check of forks and of the turn that adopts the
# state retained after a fork. Copies snapshot meta + dump index out of the container for pairing checks.
# run: flock $L sg docker -c "DEBUGDUMP=1 bash verify_run_gpu.sh verify_stage_F.sh"
set -x
S=../../results/agentic/trajectories_short_production.json
O=verify_results/F; mkdir -p $O
python3 verify_state_fork.py $S $O/cases.json --sessions 0,1,2,3,4,5
docker exec $NAME python3 /w/state_check_analyze.py /tmp/ssdump /w/$O/cases.json > $O/state_check.json
docker cp $NAME:/tmp/ssdump/meta.jsonl $O/meta.jsonl
docker exec $NAME ls /tmp/ssdump > $O/dump_index.txt
docker logs $NAME 2>&1 | grep "Session state cache:" | tail -1
