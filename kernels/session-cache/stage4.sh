# Stage 4: server with --prefix-match-unit 16 (native fine-grained prompt-tail checkpoints).
set -x
T=../../results/agentic/trajectories_long_production.json
S=../../results/agentic/trajectories_short_production.json
O=results/s4; mkdir -p $O
python3 session_bench.py $T $O/long_c1_nosess.json --sessions 0,2,4,6 --start 20 --turns 20 --consistent --tag p1n
python3 session_bench.py $T $O/long_c1_sess.json --sessions 0,2,4,6 --start 20 --turns 20 --consistent --session-id --tag p1s
python3 session_bench.py $T $O/long_c8_sess.json --start 20 --turns 20 --conc 8 --consistent --session-id --tag p8s
python3 session_bench.py $T $O/long_c8_nosess.json --start 20 --turns 20 --conc 8 --consistent --tag p8n
python3 session_bench.py $S $O/short_con_nat_nosess.json --sessions 0,1,2,3 --turns 12 --consistent --natural-stop --tag pcnn
