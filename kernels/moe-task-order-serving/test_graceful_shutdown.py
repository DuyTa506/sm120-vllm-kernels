"""CPU child-process test: verify socket ownership, reject stale identity, signal exact pidfd."""
import copy,os,socket,subprocess,sys,tempfile,time
from pathlib import Path
from graceful_shutdown import discover,signal_api
with tempfile.TemporaryDirectory() as temp:
    script=Path(temp)/'vllm'
    script.write_text('import socket,sys,time\ns=socket.socket();s.bind(("127.0.0.1",int(sys.argv[-1])));s.listen();time.sleep(30)\n')
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1',0));port=reserve.getsockname()[1]
    child=subprocess.Popen([sys.executable,str(script),'serve','test-owned-cpu',str(port)])
    try:
        for _ in range(100):
            try:record=discover(port);break
            except AssertionError:time.sleep(.01)
        else:raise AssertionError('test listener absent')
        assert record['pid']==child.pid
        changed=copy.deepcopy(record);changed['start_ticks']='0'
        try:signal_api(changed)
        except AssertionError:pass
        else:raise AssertionError('stale identity accepted')
        assert child.poll() is None
        result=signal_api(record);assert result['nsys_pid1_signalled'] is False
        assert child.wait(timeout=3)==-15
    finally:
        if child.poll() is None:child.terminate();child.wait(timeout=3)
print('owned API pidfd signal works; stale identity rejected without killing child')
