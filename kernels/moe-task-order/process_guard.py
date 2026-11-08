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
    capture = here.parent / "moe-nvfp4/data/moe_inputs.pt"
    assert capture.stat().st_size > 0, "mount actual main-checkout capture readonly"
    inputs = torch.load(capture, map_location="cpu", weights_only=True)["hidden"]
    for layer in (0, 19, 39):
        assert tuple(inputs[layer].shape) == (8192, 2048)
    assert not torch.cuda.is_initialized()
    sources = [here / "micro.py", here / "gated_long_first.py", here / "process_guard.py"]
    pkg = Path(importlib.metadata.distribution("vllm").locate_file("vllm"))
    sources += [pkg / "model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py"]
    import importlib.util
    # Package metadata gives the source root without importing a GPU backend.
    fi=Path(importlib.metadata.distribution("flashinfer-python").locate_file("flashinfer"))
    native=fi / "fused_moe/cute_dsl/blackwell_sm12x/_moe_dynamic/gated.py"
    manifest=json.loads((here / "source_manifest.json").read_text())
    assert hashlib.sha256(native.read_bytes()).hexdigest()==manifest["baseline_sha256"]
    assert hashlib.sha256((here / "gated_long_first.py").read_bytes()).hexdigest()==manifest["candidate_sha256"]
    sources.append(native)
    print(json.dumps(dict(import_check=import_check, capture=str(capture), capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
                         layers={str(k):list(v.shape) for k,v in inputs.items()}, cuda_initialized=False,
                         versions={k:importlib.metadata.version(k) for k in ("torch", "vllm", "flashinfer-python")},
                         sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})))
elif mode == "run":
    command = [sys.executable, str(here / "run_quad.py"), str(output / "measurement")]
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
        if record["state"] == "cleanup_failed":
            raise SystemExit(8)
else:
    raise SystemExit("invalid mode")
