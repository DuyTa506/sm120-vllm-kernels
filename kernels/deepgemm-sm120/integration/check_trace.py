"""Require live captured DG kernels when enabled, and M48/64 replay NVTX evidence."""
import json,sqlite3,sys
path,profile=sys.argv[1:]
with sqlite3.connect(path) as db:
    rows=db.execute('SELECT s.value,k.graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName').fetchall()
    selected=[graph for name,graph in rows if 'EpilogueRowChannel' in name]
    columns={row[1] for row in db.execute('PRAGMA table_info(NVTX_EVENTS)')}
    observed=[]
    if 'text' in columns:
        observed += [r[0] for r in db.execute("SELECT text FROM NVTX_EVENTS WHERE text LIKE 'DG_FULL_GRAPH_REPLAY_M%'")]
    if 'textId' in columns:
        observed += [r[0] for r in db.execute("SELECT s.value FROM NVTX_EVENTS n JOIN StringIds s ON s.id=n.textId WHERE s.value LIKE 'DG_FULL_GRAPH_REPLAY_M%'")]
    replay_markers=sorted(set(s for s in observed if s in ('DG_FULL_GRAPH_REPLAY_M48','DG_FULL_GRAPH_REPLAY_M64')))
    correlated=db.execute("""
        SELECT COALESCE(n.text,ns.value), count(*)
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id=k.demangledName
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId
        JOIN NVTX_EVENTS n ON n.globalTid=r.globalTid AND r.start>=n.start AND r.end<=n.end
        LEFT JOIN StringIds ns ON ns.id=n.textId
        WHERE k.graphNodeId>0 AND s.value LIKE '%EpilogueRowChannel%'
          AND COALESCE(n.text,ns.value) IN ('DG_FULL_GRAPH_REPLAY_M48','DG_FULL_GRAPH_REPLAY_M64')
        GROUP BY COALESCE(n.text,ns.value)
    """).fetchall()
assert rows,'Empty GPU trace' 
if profile=='on':
    assert any(x is not None and x>0 for x in selected),'Missing captured fused GPU kernel'
    assert correlated,'Fused GPU kernels not correlated to live M48/64 graph launch'
else:assert not selected,'Disabled candidate executed'
assert replay_markers,'No live M48/64 CUDA graph replay marker; requested workload inactive'
print(json.dumps({'profile':profile,'fused_kernels':len(selected),'fused_graph_nodes':sum(x is not None and x>0 for x in selected),'live_replay_markers':replay_markers,'fused_kernels_correlated_to_live_replay':dict(correlated),'passed':True}))
