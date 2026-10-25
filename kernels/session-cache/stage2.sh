# Stage 2: long sessions, consistent replay, A/B on one server: 8 and 32 agents.
set -x
T=../../results/agentic/trajectories_long_production.json
O=results/s2; mkdir -p $O
python3 session_bench.py $T $O/long_c8_sess.json --start 20 --turns 20 --conc 8 --consistent --session-id --tag l8s
python3 session_bench.py $T $O/long_c8_nosess.json --start 20 --turns 20 --conc 8 --consistent --tag l8n
python3 session_bench.py $T $O/long_c32_nosess.json --start 20 --turns 10 --conc 32 --copies 4 --consistent --tag l32n
python3 session_bench.py $T $O/long_c32_sess.json --start 20 --turns 10 --conc 32 --copies 4 --consistent --session-id --tag l32s
