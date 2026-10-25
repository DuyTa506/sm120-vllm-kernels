# Stage 1b: with the stop clamp. Diagnostics of recorded-prompt mismatch, correctness, short A/B, long c=1 A/B.
set -x
S=../../results/agentic/trajectories_short_production.json
L=../../results/agentic/trajectories_long_production.json
O=results/s1b; mkdir -p $O
python3 session_bench.py $S $O/short_rec_nat_sess.json --sessions 0,1,2,3 --turns 10 --natural-stop --session-id --tag rns
python3 hot_vs_cold.py $S $O/hvc_short.json --sessions 1,2 --turns 10 --cmp-tokens 24
python3 session_bench.py $S $O/short_con_nat_nosess.json --sessions 0,1,2,3 --turns 12 --consistent --natural-stop --tag cnn
python3 session_bench.py $S $O/short_con_nat_sess.json --sessions 0,1,2,3 --turns 12 --consistent --natural-stop --session-id --tag cns
python3 session_bench.py $S $O/short_con_sess.json --sessions 0,1,2,3 --turns 12 --consistent --session-id --tag cs
python3 session_bench.py $L $O/long_c1_nosess.json --sessions 0,2,4,6 --start 20 --turns 20 --consistent --tag l1n
python3 session_bench.py $L $O/long_c1_sess.json --sessions 0,2,4,6 --start 20 --turns 20 --consistent --session-id --tag l1s
