"""One GPU-lock hold: correctness, then the CUDA-graph benchmarks. Output goes to results/."""
import subprocess, sys
for script, out in (("test_correctness.py", "results/correctness.txt"), ("bench_graph.py", "results/bench_graph.txt"),
                    ("bench_ablate.py", "results/bench_ablate_v2.txt")):
    with open(out, "w") as f:
        p = subprocess.run([sys.executable, script], stdout=f, stderr=subprocess.STDOUT)
    print(script, "exit", p.returncode, flush=True)
