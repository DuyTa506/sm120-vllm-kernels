"""Require actual4096 native nodes inside a live piecewise CUDA graph launch."""
import argparse,hashlib,json,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('trace',type=Path);p.add_argument('profile',choices=['off','on']);a=p.parse_args()
c=sqlite3.connect('file:'+str(a.trace)+'?mode=ro',uri=True)
rows=c.execute('select s.value,count(*),sum(k.end-k.start)/1e6,sum(case when k.graphNodeId>0 then 1 else 0 end) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName group by s.value').fetchall()
gated=[dict(name=n,count=count,sum_ms=ms,graph_nodes=nodes) for n,count,ms,nodes in rows if 'MoEGatedDynamicKernel' in n]
assert gated,'No actual native gated execution in trace'
long=[r for r in gated if 'LongFirst' in r['name']]
assert bool(long)==(a.profile=='on'),'Candidate class activation mismatch'
if a.profile=='on':assert len(long)==len(gated),'Mixed native baseline/candidate class execution'
assert all(r['graph_nodes']==r['count'] for r in gated),'Native target ran eagerly: screen is not4096 graph-prefill evidence'
sql="""SELECT s.value,count(*),sum(k.end-k.start)/1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId JOIN StringIds rs ON rs.id=r.nameId JOIN NVTX_EVENTS n ON n.globalTid=r.globalTid AND r.start>=n.start AND r.end<=n.end LEFT JOIN StringIds ns ON ns.id=n.textId WHERE k.graphNodeId>0 AND rs.value LIKE 'cudaGraphLaunch%' AND COALESCE(n.text,ns.value)='TASK_ORDER_LIVE_PIECEWISE_M4096' AND s.value LIKE '%MoEGatedDynamicKernel%' GROUP BY s.value"""
linked=[dict(name=name,count=count,sum_ms=ms) for name,count,ms in c.execute(sql)]
assert sum(x['count'] for x in linked)==sum(x['count'] for x in gated),'Native nodes not fully tied to live4096 piecewise graph launches'
assert sum(x['count'] for x in linked)>=40,'Expected at least40 backbone native MoE layers'
print(json.dumps(dict(profile=a.profile,passed=True,kernels=gated,live4096_graph_kernels=linked,trace_sha256=hashlib.sha256(a.trace.read_bytes()).hexdigest(),scope='Live4096 piecewise graph target-node evidence; post-workload queue separately proves retained actual nonidentity descriptor order')))
