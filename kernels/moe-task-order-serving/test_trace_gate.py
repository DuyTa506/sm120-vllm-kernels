"""Reject unused labels, eager kernels, wrong native class or unrelated graph launches."""
import sqlite3,subprocess,sys,tempfile
from pathlib import Path
checker=Path(__file__).with_name('check_trace.py')
with tempfile.TemporaryDirectory() as temp:
    p=Path(temp)/'trace.sqlite';c=sqlite3.connect(p)
    c.executescript('CREATE TABLE StringIds(id INTEGER,value TEXT);CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,demangledName INTEGER,graphNodeId INTEGER,correlationId INTEGER);CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(correlationId INTEGER,start INTEGER,end INTEGER,globalTid INTEGER,nameId INTEGER);CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);')
    c.execute('INSERT INTO StringIds VALUES(1,?)',('kernel_MoEGatedDynamicKernelLongFirst',));c.execute('INSERT INTO StringIds VALUES(2,?)',('TASK_ORDER_LIVE_PIECEWISE_M4096',))
    c.execute('INSERT INTO StringIds VALUES(3,?)',('cudaGraphLaunch_v10000',));c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(4,20,30,7,3)')
    c.executemany('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(40,50,1,1,4)',[()]*40);c.commit()
    def run(profile):return subprocess.run([sys.executable,str(checker),str(p),profile],capture_output=True).returncode
    assert run('on')!=0 # Merely present StringIds marker is insufficient.
    c.execute('INSERT INTO NVTX_EVENTS VALUES(10,35,7,NULL,2)');c.commit()
    assert run('on')==0 and run('off')!=0
    c.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphNodeId=0');c.commit();assert run('on')!=0
    c.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphNodeId=1');c.execute('UPDATE NVTX_EVENTS SET start=31');c.commit();assert run('on')!=0
    c.execute('UPDATE NVTX_EVENTS SET start=10');c.execute('UPDATE StringIds SET value=? WHERE id=1',('kernel_MoEGatedDynamicKernel',));c.commit()
    assert run('off')==0 and run('on')!=0
print('trace gate requires all native nodes captured and correlated to live4096 piecewise graph launches')
