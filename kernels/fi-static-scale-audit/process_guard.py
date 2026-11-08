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
    import torch
    import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch as dispatch
    from flashinfer.cute_dsl import fp4_common
    capture=here.parent/'moe-nvfp4/data/moe_inputs.pt'
    data=torch.load(capture,map_location='cpu',weights_only=True)
    assert data['hidden'][19].shape == (8192,2048)
    assert data['hidden'][19].dtype == torch.bfloat16
    assert not torch.cuda.is_initialized()
    assert hashlib.sha256(capture.read_bytes()).hexdigest()=='0e161c36a6f478c0205b3a7e8b388296eccafaa2d5999cf4420fcbcd0aa2f4ac'
    base=here.parent/'moe-nvfp4'
    runtime=list(map(Path,dispatch._kernel_source_files()))+[Path(fp4_common.__file__)]
    runtime += list(Path(dispatch.__file__).parent.rglob('*.py'))
    runtime += [base/p for p in ['moe_common.py','test_shared_workspace.py','verify_scalar_dispatch.py','verify_shared_workspace.py','scalar/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py']]
    pkg=Path(importlib.metadata.distribution('vllm').locate_file('vllm'))
    runtime += [pkg/'_custom_ops.py'] + list(pkg.glob('_C*.so'))
    runtime=list(dict.fromkeys(runtime))
    (output/'runtime_sources').mkdir(exist_ok=False)
    for i,f in enumerate(runtime):
        if f.suffix=='.py':shutil.copy2(f,output/'runtime_sources'/f'{i}_{f.name}')
    sys.path.insert(0,str(base))
    from moe_common import snap,get_tensor
    checkpoint=Path(snap());index=checkpoint/'model.safetensors.index.json'
    wm=json.loads(index.read_text())['weight_map']
    selected={v for k,v in wm.items() if k.startswith('model.language_model.layers.19.mlp.')}
    ck=dict(path=str(checkpoint),index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),shards={n:dict(resolved=str((checkpoint/n).resolve()),size=(checkpoint/n).stat().st_size,mtime_ns=(checkpoint/n).stat().st_mtime_ns) for n in sorted(selected)},scope='index content hash and shard filesystem identity; not full shard content hashes')
    scale_keys=[f'model.language_model.layers.19.mlp.experts.{e}.{projection}.input_global_scale' for e in range(256) for projection in ('gate_proj','down_proj')]
    ck['input_scale_sha256']={k:hashlib.sha256(get_tensor(k).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k in scale_keys}
    (output/'checkpoint_identity.json').write_text(json.dumps(ck,indent=2))
    assert not torch.cuda.is_initialized()
    record=dict(capture=str(capture),capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),runtime_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in runtime},capture_key='hidden[19]',shape=[8192,2048],cuda_initialized=False)
    (output/'runtime_manifest.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record))
elif mode == "verify":
    record=json.loads((output/'runtime_manifest.json').read_text())
    for name,sha in record['runtime_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
    assert hashlib.sha256(Path(record['capture']).read_bytes()).hexdigest()==record['capture_sha256']
    print(json.dumps(dict(passed=True,runtime_files=len(record['runtime_sha256']),capture_unchanged=True)))
elif mode == "run":
    experiment=os.environ.get('AUDIT_EXPERIMENT','probe');assert experiment in ('probe','buckets')
    command=[sys.executable,str(here/('probe.py' if experiment=='probe' else 'bucket_probe.py')),'--output',str(output/'measurement.json')]
    if experiment=='buckets':
        tokens=int(os.environ['AUDIT_TOKENS']);assert tokens in (4,8)
        command += ['--tokens',str(tokens)]
    env=dict(os.environ,FLASHINFER_WORKSPACE_BASE=str(output/'cache/fi'),CUTE_DSL_CACHE_DIR=str(output/'cache/cute'),TORCHINDUCTOR_CACHE_DIR=str(output/'cache/inductor'),TRITON_CACHE_DIR=str(output/'cache/triton'))
    child = subprocess.Popen(command, start_new_session=True, env=env)
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
