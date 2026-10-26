# Stage 7: --prefix-match-unit 16 server again, A/B order reversed vs stage 4, plus engine step counts
# and server-side prefill time (metrics_delta) to explain the c=8 TTFT result of stage 4.
set -x
T=../../results/agentic/trajectories_long_production.json
S=../../results/agentic/trajectories_short_production.json
O=results/s7; mkdir -p $O
python3 session_bench.py $T $O/long_c8_nosess.json --start 20 --turns 20 --conc 8 --consistent --tag q8n
python3 session_bench.py $T $O/long_c8_sess.json --start 20 --turns 20 --conc 8 --consistent --session-id --tag q8s
python3 session_bench.py $S $O/short_con_nat_sess.json --sessions 0,1,2,3 --turns 12 --consistent --natural-stop --session-id --tag qcns
python3 session_bench.py $S $O/short_con_nat_nosess.json --sessions 0,1,2,3 --turns 12 --consistent --natural-stop --tag qcnn
python3 session_bench.py $T $O/long_c1_sess.json --sessions 0,2,4,6 --start 20 --turns 20 --consistent --session-id --tag q1s
python3 session_bench.py $T $O/long_c1_nosess.json --sessions 0,2,4,6 --start 20 --turns 20 --consistent --tag q1n
