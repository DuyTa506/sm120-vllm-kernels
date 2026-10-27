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
import sys
import time
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


def inspect_owned_container(name, run_id, *, run=subprocess.run):
    try:
        reply = run(["docker", "inspect", name], capture_output=True, text=True, timeout=15)
        if reply.returncode:
            absent = "no such object" in reply.stderr.lower() or "no such container" in reply.stderr.lower()
            return {"absent": absent, "verified_stopped": absent, "error": reply.stderr.strip()}
        records = json.loads(reply.stdout)
        if len(records) != 1: raise ValueError("Expected one exact container")
        record = records[0]
        if (record["Name"] != "/" + name
                or record["Config"]["Labels"].get("gdn.state_probe_owner") != run_id):
            raise ValueError("Container ownership does not match; no mutation allowed")
        state = record["State"]
        stopped = (state.get("Running") is False and state.get("Pid") == 0
                   and state.get("Restarting") is False and state.get("Status") in ("exited", "dead"))
        return {"absent": False, "owned": True, "id": record["Id"],
                "running": state.get("Running"), "pid": state.get("Pid"),
                "restarting": state.get("Restarting"), "status": state.get("Status"), "verified_stopped": stopped}
    except Exception as exc:
        return {"absent": False, "verified_stopped": False, "error": str(exc)}


def cleanup_owned_container(name, run_id, *, run=subprocess.run):
    report = {"container": name, "actions": [], "observations": []}
    def inspect():
        state = inspect_owned_container(name, run_id, run=run)
        report["observations"].append(state)
        return state
    def action(command, timeout):
        try:
            reply = run(command, capture_output=True, text=True, timeout=timeout)
            report["actions"].append({"argv": command, "returncode": reply.returncode,
                                      "stdout": reply.stdout.strip(), "stderr": reply.stderr.strip()})
        except (OSError, subprocess.TimeoutExpired) as exc:
            report["actions"].append({"argv": command, "returncode": None, "error": str(exc)})
    state = inspect()
    if not state["verified_stopped"] and state.get("owned"):
        action(["docker", "stop", "--time", "30", state["id"]], 45)
        state = inspect()
        if not state["verified_stopped"] and state.get("owned"):
            # Rechecked label/name above. Use the immutable owned ID, so a name
            # replacement between inspection and removal cannot target a stranger.
            action(["docker", "rm", "-f", state["id"]], 30)
            state = inspect()
    report["verified_stopped"] = state["verified_stopped"]
    report["status"] = "verified_stopped" if state["verified_stopped"] else "cleanup_failed"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--run-gpu", action="store_true")
    group.add_argument("--preflight", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--hf-cache", type=Path, default=Path("/opt/intramind/hf-cache"))
    parser.add_argument("--output-dir", type=Path, default=HERE / "results/packed_state_probe_3")
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
    previous_sigint = signal.getsignal(signal.SIGINT)
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
        cleanup = None
        recovery_required = False
        if process is not None:
            def retain_lock(signum, frame):
                print("Owned container cleanup is unverified or in progress; GPU lock retained until it is stopped.", file=sys.stderr, flush=True)
            signal.signal(signal.SIGTERM, retain_lock)
            signal.signal(signal.SIGINT, retain_lock)
            cleanup = cleanup_owned_container(name, run_id)
            plan["cleanup"] = cleanup
            plan["gpu_slot_release_confirmed"] = bool(args.run_gpu and cleanup["verified_stopped"])
            record_path.write_text(json.dumps(plan, indent=2) + "\n")
            if not cleanup["verified_stopped"]:
                plan["probe_exit_code"] = plan.get("exit_code")
                plan["exit_code"] = 1
                record_path.write_text(json.dumps(plan, indent=2) + "\n")
            if not cleanup["verified_stopped"] and lock is not None:
                recovery_required = True
                plan["cleanup_recovery_required"] = True
                record_path.write_text(json.dumps(plan, indent=2) + "\n")
                retain_lock(None, None)
                # No more destructive retries. Keep the lock while an operator
                # resolves this exact owned container, then verify its state.
                while not cleanup["verified_stopped"]:
                    time.sleep(10)
                    state = inspect_owned_container(name, run_id)
                    cleanup["observations"].append(state)
                    cleanup["verified_stopped"] = state["verified_stopped"]
                    if state["verified_stopped"]:
                        cleanup["status"] = "recovered_after_cleanup_failure"
                    plan["gpu_slot_release_confirmed"] = state["verified_stopped"]
                    record_path.write_text(json.dumps(plan, indent=2) + "\n")
            if process.poll() is None and cleanup["verified_stopped"]:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()  # Only this runner's Docker CLI child; owned container is stopped.
                    process.wait(timeout=10)
        signal.signal(signal.SIGTERM, previous)
        signal.signal(signal.SIGINT, previous_sigint)
        if lock is not None and (cleanup is None or cleanup["verified_stopped"]):
            lock.close()
        if cleanup is not None and (not cleanup["verified_stopped"] or recovery_required):
            raise RuntimeError("Owned container cleanup failed; see recorded cleanup state/recovery")



if __name__ == "__main__": main()
