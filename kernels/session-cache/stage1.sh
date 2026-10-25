# Stage 1: smoke + correctness on short sessions, then a short-session A/B (same server).
set -x
T=../../results/agentic/trajectories_short_production.json
O=results; mkdir -p $O
python3 hot_vs_cold.py $T $O/hvc_short_s0.json --sessions 0 --turns 10 --cmp-tokens 24
python3 session_bench.py $T $O/s1_short_rec_nosess.json --sessions 0,1,2,3 --turns 12 --tag rec-nosess
python3 session_bench.py $T $O/s1_short_rec_sess.json --sessions 0,1,2,3 --turns 12 --session-id --tag rec-sess
python3 session_bench.py $T $O/s1_short_con_nosess.json --sessions 0,1,2,3 --turns 12 --consistent --tag con-nosess
python3 session_bench.py $T $O/s1_short_con_sess.json --sessions 0,1,2,3 --turns 12 --consistent --session-id --tag con-sess
