"""CPU evidence-gate tests: unused labels/eager kernels must never prove live capture."""
from pathlib import Path
import sqlite3,subprocess,sys,tempfile
check=Path(__file__).with_name('check_trace.py')
with tempfile.TemporaryDirectory() as temp:
    p=Path(temp)/'trace.sqlite'
    with sqlite3.connect(p) as db:
        db.executescript('CREATE TABLE StringIds(id INTEGER,value TEXT); CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(demangledName INTEGER,graphNodeId INTEGER,correlationId INTEGER); CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER); CREATE TABLE NVTX_EVENTS(text TEXT,textId INTEGER,start INTEGER,end INTEGER,globalTid INTEGER);')
        db.execute('INSERT INTO StringIds VALUES(1,?)',('deep_gemm::EpilogueRowChannel',))
        db.execute('INSERT INTO StringIds VALUES(2,?)',('DG_FULL_GRAPH_REPLAY_M48',))
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(1,42,8)')
        db.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(110,120,5,8)')
    def expect(ok,profile='on'):
        result=subprocess.run([sys.executable,str(check),str(p),profile],capture_output=True)
        assert (result.returncode==0)==ok,result.stdout+result.stderr
    expect(False)  # A label in StringIds without a live NVTX event is insufficient.
    with sqlite3.connect(p) as db:db.execute('INSERT INTO NVTX_EVENTS VALUES(NULL,2,100,130,5)')
    expect(True)
    with sqlite3.connect(p) as db:db.execute('UPDATE NVTX_EVENTS SET start=200,end=230')
    expect(False)  # Unrelated candidate kernels and live markers must not satisfy the gate.
    with sqlite3.connect(p) as db:db.execute('UPDATE NVTX_EVENTS SET start=100,end=130')
    expect(False,'off')  # Disabled control must not execute candidate kernels.
    with sqlite3.connect(p) as db:db.execute('UPDATE CUPTI_ACTIVITY_KIND_KERNEL SET graphNodeId=0')
    expect(False)  # Eager candidate execution does not prove captured graph activation.
print('Trace evidence gate checks passed')
