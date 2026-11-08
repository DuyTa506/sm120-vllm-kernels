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
    imported = subprocess.run([sys.executable, str(here / "import_preflight.py")], capture_output=True, text=True, check=True)
    import_check = json.loads(imported.stdout.strip().splitlines()[-1])
    import torch
    capture = here.parent / "moe-task-order/results/task_order_4096_1/measurement/a1/l19_m4096_real.pt"
    data = torch.load(capture, map_location="cpu", weights_only=True)
    assert tuple(data["input"].shape)==(4096,2048)
    assert tuple(data["ids"].shape)==(4096,8)
    assert not torch.cuda.is_initialized()
    manifest=json.loads((here / "source_manifest.json").read_text())
    for variant, record in manifest.items():
        p=here/f"gated_{variant}_diagnostic.py"
        assert hashlib.sha256(p.read_bytes()).hexdigest()==record["diagnostic_sha256"]
    fi=Path(importlib.metadata.distribution("flashinfer-python").locate_file("flashinfer"))
    runtime=[fi/"fused_moe/cute_dsl/blackwell_sm12x/_moe_dynamic/gated.py",fi/"fused_moe/cute_dsl/blackwell_sm12x/moe_dynamic_kernel.py",fi/"fused_moe/cute_dsl/blackwell_sm12x/moe_dispatch.py",here.parent/"moe-nvfp4/scalar/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py"]
    assert hashlib.sha256(runtime[0].read_bytes()).hexdigest()==manifest["baseline"]["source_sha256"]
    (output/"runtime_sources").mkdir(parents=True,exist_ok=True)
    import shutil
    for index,p in enumerate(runtime):shutil.copy2(p,output/"runtime_sources"/f"{index}_{p.name}")
    print(json.dumps(dict(import_check=import_check,capture=str(capture),capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),cuda_initialized=False,source_manifest=manifest,runtime_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime})))
elif mode == "run":
    variant=os.environ["TASK_DIAG_VARIANT"]
    assert variant in ("baseline","long_first")
    columns=int(os.environ.get('TASK_DIAG_COLUMNS','128'))
    assert columns in (128,2048)
    command = [sys.executable, str(here / "diagnose.py"), "--variant", variant, "--columns", str(columns),
               "--capture", str(here.parent / "moe-task-order/results/task_order_4096_1/measurement/a1/l19_m4096_real.pt"),
               "--output", str(output / "measurement")]
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
        print(json.dumps(record))
        if Path(f"/proc/{record['pid']}").exists():
            raise SystemExit("owned PID still present after cleanup")
        if record["state"] == "cleanup_failed":
            raise SystemExit(8)
else:
    raise SystemExit("invalid mode")
