# Verify stage D (production + session cache, default TTL/MAX): capacity. Long c=32 replay before and after
# 250 idle sessions pin ~1000 blocks (default MAX 256 sessions, evicted only below 2 % free).
# run: flock $L sg docker -c "bash verify_run_gpu.sh verify_stage_D.sh"
set -x
L=../../results/agentic/trajectories_long_production.json
O=verify_results/D; mkdir -p $O
python3 session_bench.py $L $O/c32_before.json --start 20 --turns 4 --copies 4 --conc 32 --consistent --session-id --tag vd1
docker logs $NAME 2>&1 | grep "Session state cache:" | tail -1
python3 verify_junk_sessions.py 250 16
docker logs $NAME 2>&1 | grep "Session state cache:" | tail -1
python3 session_bench.py $L $O/c32_after.json --start 20 --turns 4 --copies 4 --conc 32 --consistent --session-id --tag vd2
docker logs $NAME 2>&1 | grep "Session state cache:" | tail -1
