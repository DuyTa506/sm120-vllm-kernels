"""Require actual native-direct graph kernels correlated to live actual M3 replay."""
import json,sqlite3,sys
path,profile=sys.argv[1:]
with sqlite3.connect(path) as db:
 rows=db.execute('SELECT s.value,k.graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName').fetchall()
 native=[g for name,g in rows if 'MoEDirectMicroKernel' in name]
 sanitizer=[g for name,g in rows if '_sanitize' in name]
 markers=[r[0] for r in db.execute("SELECT COALESCE(n.text,s.value) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId WHERE COALESCE(n.text,s.value) LIKE 'FI_DIRECT_LIVE_GRAPH_M%'")]
 correlated=db.execute("""SELECT COALESCE(n.text,ns.value),count(*)
 FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName
 JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId
 JOIN NVTX_EVENTS n ON n.globalTid=r.globalTid AND r.start>=n.start AND r.end<=n.end
 LEFT JOIN StringIds ns ON ns.id=n.textId
 WHERE k.graphNodeId>0 AND s.value LIKE '%MoEDirectMicroKernel%'
 AND COALESCE(n.text,ns.value) = 'FI_DIRECT_LIVE_GRAPH_M3'
 GROUP BY COALESCE(n.text,ns.value)""").fetchall()
assert rows and markers,'no actual workload GPU graph evidence'
if profile=='on':
 assert correlated,'no native direct graph kernel correlated to eligible live shape'
 assert any(g and g>0 for g in sanitizer),'sanitizer absent from CUDA graph'
else:assert not native and not sanitizer,'candidate kernel executed with flag off'
print(json.dumps(dict(profile=profile,native_kernel_events=len(native),sanitizer_events=len(sanitizer),
 live_markers=sorted(set(markers)),native_correlated=dict(correlated),passed=True)))
