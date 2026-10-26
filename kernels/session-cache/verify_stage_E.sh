# Verify stage E (production + session cache): concurrent same-session adopters with different continuations
# (copy-on-write), fork vs solo vs cold-vs-cold.
# run: flock $L sg docker -c "bash verify_run_gpu.sh verify_stage_E.sh"
set -x
S=../../results/agentic/trajectories_short_production.json
O=verify_results/E; mkdir -p $O
python3 verify_fork.py $S $O/fork.json --sessions 0,1,2,3,4,5,6,7 --starts 0,1,2,3,4,5,6,7
