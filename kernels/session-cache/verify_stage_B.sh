# Verify stage B (production config, session cache on): reproduce long c=1 A/B, equal-workload short A/B
# (ignore_eos, fixed completion lengths), full-generation hot vs cold vs cold with forks at conc 4.
# run: flock $L sg docker -c "bash verify_run_gpu.sh verify_stage_B.sh"
set -x
S=../../results/agentic/trajectories_short_production.json
L=../../results/agentic/trajectories_long_production.json
O=verify_results/B; mkdir -p $O
python3 session_bench.py $L $O/long_c1_sess.json --sessions 0,2,4,6 --start 20 --turns 20 --consistent --session-id --tag vl1s
python3 session_bench.py $L $O/long_c1_nosess.json --sessions 0,2,4,6 --start 20 --turns 20 --consistent --tag vl1n
python3 session_bench.py $S $O/short_con_nosess.json --sessions 0,1,2,3 --turns 12 --consistent --tag vcn
python3 session_bench.py $S $O/short_con_sess.json --sessions 0,1,2,3 --turns 12 --consistent --session-id --tag vcs
python3 verify_hvc_full.py $S $O/hvc_full.json --sessions 0,1,2,3,4,5,6,7 --turns 8 --conc 4
