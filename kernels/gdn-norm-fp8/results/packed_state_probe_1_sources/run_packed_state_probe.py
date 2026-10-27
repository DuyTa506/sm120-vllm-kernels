#!/usr/bin/env python3
"""Own one fresh container for the state probe. Default prints the plan only.

--preflight has no GPU device access. --run-gpu takes the existing nonblocking
GPU lock and checks for active GPU compute processes before creating anything.
No serving launcher or installed source is modified.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import uuid

from probe_packed_state import HERE, ROOT, MOUNTS

LOCK = Path("/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/"
            "2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock")
PKG = "/usr/local/lib/python3.12/dist-packages/vllm"


def build_command(args, run_id):
    name = "gdn-private-state-" + run_id
    output = args.output_dir.resolve()
    relative = output.relative_to(ROOT)
    command = ["docker", "run", "--rm", "--name", name, "--label", "gdn.state_probe_owner=" + run_id,
               "--ipc=host", "--entrypoint", "python3", "-v", str(ROOT) + ":/work",
               "-v", str(args.hf_cache.resolve()) + ":/root/.cache/huggingface:ro"]
    for relative_source in MOUNTS:
        command += ["-v", str(ROOT / "build/vllm" / relative_source) + ":" + PKG + "/" + relative_source + ":ro"]
    for key, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONPATH": "/work/kernels/gdn-norm-fp8",
                       "VLLM_GDN_INTERNAL_CKPT": "1", "VLLM_GDN_FUSED_CONV": "1", "VLLM_SM120_FP8_PREFILL": "1",
                       "VLLM_SM120_FP8_PREFILL_NOGLUE": "1", "VLLM_FI_KERNEL_PAGES": "16",
                       "VLLM_GDN_NORM_FP8_FUSION": "0"}.items():
        command += ["-e", key + "=" + value]
    if args.run_gpu:
        command += ["--gpus", "device=" + str(args.gpu),
                    "-v", str(ROOT / ".cache/vllm/gdn-private-state-probe") + ":/root/.cache/vllm",
                    "-v", str(ROOT / ".cache/flashinfer") + ":/root/.cache/flashinfer"]
    else:
        command += ["-e", "CUDA_VISIBLE_DEVICES=", "-e", "NVIDIA_VISIBLE_DEVICES=void"]
    command += ["vllm/vllm-openai:v0.32.0", "/work/kernels/gdn-norm-fp8/probe_packed_state.py",
                "--run-gpu" if args.run_gpu else "--preflight", "--output-dir", str(Path("/work") / relative)]
    return name, command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-gpu", action="store_true")
    group.add_argument("--preflight", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--hf-cache", type=Path, default=Path("/opt/intramind/hf-cache"))
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/packed_state_probe_1")
    parser.add_argument("--lock", type=Path, default=Path(os.environ.get("GPU_LOCK", str(LOCK))))
    args = parser.parse_args()
    run_id = uuid.uuid4().hex
    name, command = build_command(args, run_id)
    plan = {"container": name, "argv": command, "gpu_launch": args.run_gpu,
            "lock": str(args.lock), "output_dir": str(args.output_dir.resolve())}
    if not args.run_gpu and not args.preflight:
        print(json.dumps({"status": "DRY_RUN", **plan}, indent=2)); return
    if args.run_gpu and args.output_dir.exists(): raise FileExistsError(args.output_dir)
    lock = None; process = None
    def terminate(signum, frame): raise SystemExit(128 + signum)
    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        if args.run_gpu:
            lock = args.lock.open("a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            active = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu),
                "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
            if active: raise RuntimeError("GPU has active compute processes; no container started")
            (ROOT / ".cache/vllm/gdn-private-state-probe").mkdir(parents=True, exist_ok=True)
            (ROOT / ".cache/flashinfer").mkdir(parents=True, exist_ok=True)
        if not args.hf_cache.is_dir(): raise FileNotFoundError(args.hf_cache)
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        record_path = args.output_dir.with_name(args.output_dir.name + ".runner.json")
        if record_path.exists(): raise FileExistsError(record_path)
        record_path.write_text(json.dumps(plan, indent=2) + "\n")
        process = subprocess.Popen(command)
        code = process.wait()
        plan["exit_code"] = code
        record_path.write_text(json.dumps(plan, indent=2) + "\n")
        if code: raise SystemExit(code)
    finally:
        # Inspect only our random name, and require its ownership label before stop.
        if process is not None:
            check = subprocess.run(["docker", "inspect", "--format",
                '{{index .Config.Labels "gdn.state_probe_owner"}}', name],
                capture_output=True, text=True)
            if check.returncode == 0 and check.stdout.strip() == run_id:
                subprocess.run(["docker", "stop", "--time", "60", name], check=False, timeout=75)
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
        signal.signal(signal.SIGTERM, previous)
        if lock is not None: lock.close()


if __name__ == "__main__": main()
