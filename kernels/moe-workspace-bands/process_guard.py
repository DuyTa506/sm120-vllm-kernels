"""CPU preflight and exact subprocess-group lifetime for the isolated GPU test."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

mode, output = sys.argv[1], Path(sys.argv[2])
here = Path(__file__).resolve().parent
marker = output / "process.json"

def start_ticks(pid):
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]

if mode == "preflight":
    import shutil
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="")
    imported=subprocess.run([sys.executable,str(here/"test_workspace_adapter.py")],env=env,capture_output=True,text=True,check=True)
    import torch
    capture=here.parent/"moe-task-order/results/task_order_4096_1/measurement/a1/l19_m4096_real.pt"
    data=torch.load(capture,map_location="cpu",weights_only=True)
    capture_pin=json.loads((here/'capture_manifest.json').read_text())
    assert hashlib.sha256(capture.read_bytes()).hexdigest()==capture_pin['capture_sha256']
    for key,spec in capture_pin['keys'].items():
        assert list(data[key].shape)==spec['shape'] and str(data[key].dtype)==spec['dtype'],key
    sidecar=capture.with_name('applicability_l19_m4096_real.json')
    assert hashlib.sha256(sidecar.read_bytes()).hexdigest()==capture_pin['sidecar_sha256']
    metadata=json.loads(sidecar.read_text())
    for key in ('layer','tokens','pattern'):assert metadata[key]==capture_pin[key]
    assert torch.bincount(data['ids'].flatten().long(),minlength=256).tolist()==metadata['row_counts']
    assert not torch.cuda.is_initialized()
    base=here.parent/'moe-nvfp4'
    fi=Path(importlib.metadata.distribution('flashinfer-python').locate_file('flashinfer'))
    runtime=[fi/'fused_moe/cute_dsl/blackwell_sm12x/_moe_dynamic/gated.py',fi/'fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py',fi/'fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py']
    runtime += [base/p for p in ['moe_common.py','test_shared_workspace.py','verify_scalar_dispatch.py','verify_shared_workspace.py','scalar/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py']]
    assert hashlib.sha256(runtime[0].read_bytes()).hexdigest()=='993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445'
    pkg=Path(importlib.metadata.distribution('vllm').locate_file('vllm'))
    runtime += [pkg/'_custom_ops.py',pkg/'v1/worker/workspace.py',fi/'fused_moe/cute_dsl/b12x_moe.py'] + list(pkg.glob('_C*.so'))
    (output/'runtime_sources').mkdir(exist_ok=False)
    for i,f in enumerate(runtime):
        if f.suffix=='.py':shutil.copy2(f,output/'runtime_sources'/f'{i}_{f.name}')
    record=dict(capture=str(capture),capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),runtime_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in runtime+[sidecar]},capture_pin=capture_pin,import_check={'cpu_tests_passed':True,'stderr':imported.stderr},cuda_initialized=False)
    (output/'runtime_manifest.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record))
elif mode == "verify":
    record=json.loads((output/'runtime_manifest.json').read_text())
    for name,sha in record['runtime_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
    assert hashlib.sha256(Path(record['capture']).read_bytes()).hexdigest()==record['capture_sha256']
    print(json.dumps(dict(passed=True,runtime_files=len(record['runtime_sha256']),capture_unchanged=True)))
elif mode == "run":
    experiment='micrograph'
    command=[sys.executable,str(here/(experiment+'.py')),'--capture',str(here.parent/'moe-task-order/results/task_order_4096_1/measurement/a1/l19_m4096_real.pt'),'--output',str(output/'measurement')]
    child = subprocess.Popen(command, start_new_session=True)
    record = dict(pid=child.pid, pgid=os.getpgid(child.pid), start_ticks=start_ticks(child.pid), command=command,
                  started=time.time(), state="running")
    marker.write_text(json.dumps(record, indent=2))
    code = child.wait()
    record.update(state="exited", exit_code=code, ended=time.time())
    marker.write_text(json.dumps(record, indent=2))
    raise SystemExit(code)
elif mode == "stop":
    if not marker.exists():
        print(json.dumps(dict(state="never_started")))
    else:
        record=json.loads(marker.read_text())
        if record["state"] == "running":
            try:
                assert start_ticks(record["pid"]) == record["start_ticks"], "PID reused; refuse signal"
                assert os.getpgid(record["pid"]) == record["pgid"] == record["pid"]
                os.killpg(record["pgid"], signal.SIGTERM)
                for _ in range(50):
                    if not Path(f"/proc/{record['pid']}").exists():
                        break
                    time.sleep(0.1)
                if Path(f"/proc/{record['pid']}").exists():
                    assert start_ticks(record["pid"]) == record["start_ticks"]
                    os.killpg(record["pgid"], signal.SIGKILL)
                    for _ in range(50):
                        if not Path(f"/proc/{record['pid']}").exists():
                            break
                        time.sleep(0.1)
                record.update(state="terminated" if not Path(f"/proc/{record['pid']}").exists() else "cleanup_failed", ended=time.time())
            except FileNotFoundError:
                record.update(state="process_absent_without_terminal_record")
        # Verify descendants of the exact owned session/group, including a
        # compiler child surviving its leader. Never stop the dev container.
        def members():
            found=[]
            for entry in Path('/proc').iterdir():
                if not entry.name.isdigit():continue
                try:
                    fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
                    if int(fields[2])==record['pgid']:
                        assert int(fields[19])>=int(record['start_ticks'])
                        found.append(int(entry.name))
                except FileNotFoundError:pass
            return found
        remaining=members()
        if remaining:
            leader=Path(f"/proc/{record['pid']}")
            if leader.exists():assert start_ticks(record['pid'])==record['start_ticks'],'PID reused; refuse group signal'
            for sig in (signal.SIGTERM,signal.SIGKILL):
                try:os.killpg(record['pgid'],sig)
                except ProcessLookupError:break
                for _ in range(50):
                    if not members():break
                    time.sleep(.1)
                if not members():break
        record['remaining_group_pids']=members()
        print(json.dumps(record))
        if record['remaining_group_pids']:raise SystemExit('owned group still present after cleanup')
        if Path(f"/proc/{record['pid']}").exists():
            raise SystemExit("owned PID still present after cleanup")
        if record["state"] == "cleanup_failed":
            raise SystemExit(8)
else:
    raise SystemExit("invalid mode")
