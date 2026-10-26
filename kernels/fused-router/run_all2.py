"""One GPU-lock hold: cluster feasibility probe, correctness (split-K, then cluster), the CUDA-graph benchmarks, prefetch benchmark."""
import subprocess, sys
steps = (("./cluster_probe", "results/cluster_probe.txt"),
         ("test_correctness.py", "results/correctness.txt"),
         ("test_cluster.py", "results/correctness_cluster.txt"),
         ("bench_graph.py", "results/bench_graph.txt"),
         ("bench_prefetch.py", "results/bench_prefetch.txt"))
for script, out in steps:
    cmd = [script] if script.startswith("./") else [sys.executable, script]
    with open(out, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(script, "exit", p.returncode, flush=True)
