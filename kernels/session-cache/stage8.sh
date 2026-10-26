# Stage 8: direct state check (DEBUGDUMP=1 server: debug/ variant files). Session snapshots vs cold
# prefill reference snapshots at E, E-1, E+1. Analysis runs inside the container; dumps stay in its /tmp.
set -x
S=../../results/agentic/trajectories_short_production.json
O=results/s8; mkdir -p $O
python3 state_check.py $S $O/cases.json --sessions 0,1,2,3,4,5 --turns 5
docker exec $NAME python3 /w/state_check_analyze.py /tmp/ssdump /w/$O/cases.json > $O/state_check.json
python3 -c "import json; print(json.dumps(json.load(open('$O/state_check.json'))['summary'], indent=1))"
