"""Identical host-only observation in both variants; no device reads or sync."""
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

_registered = False


def register():
    global _registered
    destination = os.environ.get("MTP_SCHEDULE_TRACE")
    if _registered or not destination:
        return
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.worker.gpu_worker import Worker
    original_load = Worker.load_model

    @functools.wraps(original_load)
    def load_model(worker, *args, **kwargs):
        result = original_load(worker, *args, **kwargs)
        runner = worker.model_runner
        proposer = getattr(runner, "speculator", None)
        # This experiment targets the existing V2 baseline; do not silently
        # enable another runner or assume installed source is the live path.
        if type(runner).__module__ != "vllm.v1.worker.gpu.model_runner":
            raise RuntimeError("MTP scheduling probe requires baseline V2 runner; no runner switch was requested")
        if proposer is None or "num_speculative_tokens" not in inspect.signature(proposer.propose).parameters:
            raise RuntimeError("active proposer does not support dynamic proposal depth")
        spec = runner.vllm_config.speculative_config
        effective = dict(method=spec.method, num_speculative_tokens=spec.num_speculative_tokens,
                         num_speculative_tokens_per_batch_size=spec.num_speculative_tokens_per_batch_size)
        from mtp_schedule_config import config
        expected = config(os.environ["MTP_SCHEDULE_PROFILE"])
        for key, value in expected.items():
            actual = json.loads(json.dumps(effective[key]))
            if actual != value:
                raise RuntimeError(f"effective speculative config differs: {key}: {actual} != {value}")
        if "num_speculative_tokens_per_batch_size" not in expected and effective["num_speculative_tokens_per_batch_size"] is not None:
            raise RuntimeError("fixed2 unexpectedly has a dynamic schedule")
        classes = {}
        for name, obj in (("runner", type(runner)), ("proposer", type(proposer)), ("propose", proposer.propose)):
            path = Path(inspect.getsourcefile(obj))
            classes[name] = dict(qualified_name=f"{obj.__module__}.{obj.__qualname__}",
                                 source=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        destination_path = Path(destination)
        destination_path.mkdir(parents=True, exist_ok=True)
        (destination_path / f"runtime-{os.getpid()}.json").write_text(json.dumps(dict(classes=classes, effective=effective), indent=2))
        return result

    Worker.load_model = load_model
    if os.environ.get("MTP_DEPTH_PROBE", "1") == "0":
        _registered = True
        return
    original = Scheduler.schedule
    source = Path(inspect.getsourcefile(Scheduler))
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    # Register in engine/frontend processes; only actual schedule calls create logs.
    handle = None

    @functools.wraps(original)
    def schedule(instance, *args, **kwargs):
        nonlocal handle
        result = original(instance, *args, **kwargs)
        if handle is None:
            handle = (root / f"scheduler-{os.getpid()}.jsonl").open("a", buffering=1)
            handle.write(json.dumps(dict(kind="source", path=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest())) + "\n")
        handle.write(json.dumps(dict(kind="step", monotonic=time.monotonic(),
                                    requests=len(result.num_scheduled_tokens),
                                    tokens=result.total_num_scheduled_tokens,
                                    scheduled_depth=result.num_spec_tokens_to_schedule)) + "\n")
        return result

    Scheduler.schedule = schedule
    _registered = True
