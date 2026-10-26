# Verify stage C (--prefix-match-unit 16 + session cache, the recommended deployment): incremental gain.
# run: flock $L sg docker -c "bash verify_run_gpu.sh verify_stage_C.sh --prefix-match-unit 16"
set -x
S=../../results/agentic/trajectories_short_production.json
L=../../results/agentic/trajectories_long_production.json
O=verify_results/C; mkdir -p $O
python3 verify_fallback.py $S $O/fallback.json --sessions 0,1,2,3,4,5 --turns 6 --back 50
python3 session_bench.py $S $O/short_con_sess.json --sessions 0,1,2,3 --turns 12 --consistent --session-id --tag vcs
python3 session_bench.py $S $O/short_con_nosess.json --sessions 0,1,2,3 --turns 12 --consistent --tag vcn
python3 session_bench.py $L $O/long_c1_nosess.json --sessions 0,2,4,6 --start 20 --turns 20 --consistent --tag vl1n
python3 session_bench.py $L $O/long_c1_sess.json --sessions 0,2,4,6 --start 20 --turns 20 --consistent --session-id --tag vl1s
python3 session_bench.py $L $O/long_c8_sess.json --start 20 --turns 20 --conc 8 --consistent --session-id --tag vl8s
python3 session_bench.py $L $O/long_c8_nosess.json --start 20 --turns 20 --conc 8 --consistent --tag vl8n
