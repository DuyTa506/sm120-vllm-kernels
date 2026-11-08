"""Require observed skinny kernels in captured graphs, not merely eager lm_head."""
import json
import sqlite3
import sys

path,profile,concurrency=sys.argv[1:]
with sqlite3.connect(path) as db:
    rows=db.execute('SELECT s.value,k.graphNodeId FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName').fetchall()
assert rows,'empty trace'
counts={}
for label,symbol in [('bf16','_skinny_bf16_gemm_kernel'),('fp8','skinny_fp8_kernel')]:
    selected=[graph for name,graph in rows if symbol in name]
    counts[label]=dict(all=len(selected),captured=sum(g is not None and g>0 for g in selected))
expected=dict(bf16=profile in ('bf16','both'),fp8=profile in ('fp8','both'))
for label,on in expected.items():
    if not on:assert counts[label]['all']==0,(label,'disabled kernel executed',counts)
    elif int(concurrency)==1:assert counts[label]['captured']>0,(label,'missing captured kernel activation',counts)
print(json.dumps(dict(profile=profile,concurrency=int(concurrency),counts=counts,passed=True,
                     caveat='c4 presence depends on actual padded shapes; off absence checked at both concurrencies')))
