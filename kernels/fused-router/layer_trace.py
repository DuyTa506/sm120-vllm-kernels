"""Print the kernel sequence of one decode step (target verify graph) from a node-level nsys trace,
to see what runs between the hidden state and the MoE grouped GEMM in each layer.
usage: python3 -I layer_trace.py <node_c1.sqlite> [step_index] [first_kernel] [count]
"""
import collections, sqlite3, sys
db = sqlite3.connect(sys.argv[1])
S = dict(db.execute("SELECT id, value FROM StringIds"))
cols = [r[1] for r in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
rows = db.execute("SELECT correlationId, demangledName, gridX, gridY, gridZ, blockX, streamId, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
by = collections.defaultdict(list)
for r in rows:
    by[r[0]].append(r)
sizes = collections.Counter(len(v) for v in by.values())
tn = max(n for n, c in sizes.items() if c >= 20)
steps = [v for v in by.values() if len(v) == tn]
si = int(sys.argv[2]) if len(sys.argv) > 2 else len(steps) // 2
k0 = int(sys.argv[3]) if len(sys.argv) > 3 else 0
cnt = int(sys.argv[4]) if len(sys.argv) > 4 else 60
st = sorted(steps[si], key=lambda r: r[7])
print("kernels per step", tn, "steps", len(steps), "step", si)
t0 = st[k0][7]
for i, r in enumerate(st[k0:k0 + cnt], k0):
    print(f"{i:4d} +{(r[7]-t0)/1e3:8.2f}us dur {(r[8]-r[7])/1e3:6.2f} s{r[6]} g=({r[2]},{r[3]},{r[4]}) b={r[5]} {S[r[1]][:110]}")
