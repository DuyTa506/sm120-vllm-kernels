"""Optional FlashInfer v1 JSON import and read-only tactic auditing.

No compiled-graph cache is imported. CUDA/vLLM imports occur only in install().
The supported implementation is FlashInfer 0.7.0.post1, vLLM 0.32, single GPU.
"""
import functools
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path


def tactic_entries(document):
    if not isinstance(document, dict) or not isinstance(document.get("_metadata"), dict):
        raise ValueError("autotune JSON must contain explicit environment metadata")
    entries = {k: v for k, v in document.items() if not k.startswith("_")}
    if not entries or any(not isinstance(k, str) or not k.startswith("(")
                          or not isinstance(v, list) or len(v) != 2
                          or not isinstance(v[0], str) for k, v in entries.items()):
        raise ValueError("invalid or empty FlashInfer v1 tactic table")
    if any(k.startswith("_") and k not in ("_metadata", "_generation")
           for k in document):
        raise ValueError("unsupported reserved/namespaced autotune records")
    return entries


def compare_tactics(reference, actual):
    return {"reference_entries": len(reference), "actual_entries": len(actual),
            "missing": sorted(set(reference) - set(actual)),
            "extra": sorted(set(actual) - set(reference)),
            "changed": [{"key": k, "expected": reference[k], "actual": actual[k]}
                        for k in sorted(set(reference) & set(actual)) if reference[k] != actual[k]]}


def entries_sha256(entries):
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def compiler_config_audit(cache_root):
    files = []
    for path in sorted(Path(cache_root).rglob("*.best_config")):
        data = path.read_bytes()
        config = json.loads(data)
        selected = {k: v for k, v in config.items() if k not in (
            "time_taken_ms", "triton_cache_hash", "configs_hash", "found_by_coordesc")}
        files.append({"path": str(path.relative_to(cache_root)), "basename": path.name,
                      "sha256": hashlib.sha256(data).hexdigest(), "selected": selected,
                      "selected_sha256": hashlib.sha256(json.dumps(
                          selected, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    return {"scope": "Saved Inductor .best_config files only; observed, never imported or pinned",
            "cache_root": str(cache_root), "files": files}


class AutotuneImport:
    def __init__(self, source, out, require_coverage=False, *, observe_search=True):
        if require_coverage and not observe_search:
            raise ValueError("lookup coverage requires lookup observation")
        self.source, self.out = Path(source), Path(out) / "autotune"
        self.source_bytes = self.source.read_bytes()
        self.document = json.loads(self.source_bytes)
        self.reference = tactic_entries(self.document)
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out / "source.json").write_bytes(self.source_bytes)
        self.require_coverage = require_coverage
        self.observe_search = observe_search
        self.cache_root = self.out.parent / "cache" / "autotune_import"
        self.phase, self.observed, self.misses = "warmup", {}, {}
        self.tuner = self.encode = None
        self.report = {"status": "prepared_not_loaded", "source": str(self.source),
                       "source_sha256": hashlib.sha256(self.source_bytes).hexdigest(),
                       "reference_entries": len(self.reference), "load_calls": [],
                       "reference_entries_sha256": entries_sha256(self.reference),
                       "lookup_observation_enabled": observe_search,
                       "checks": {}, "require_all_observed_keys": require_coverage,
                       "compiled_graph_cache_imported": False}
        self.write_report()

    def write_report(self):
        (self.out / "verification.json").write_text(json.dumps(self.report, indent=2) + "\n")

    def _effective_entries(self, include_winners):
        # save_configs() uses this same precedence, but also clears _dirty and
        # merges old disk data. Reading the tables avoids those side effects.
        with self.tuner._lock:
            entries = {k: [name, self.encode(tactic)]
                       for k, (name, tactic) in self.tuner._file_configs.items()}
            if include_winners:
                for key, (tactic, _) in self.tuner.profiling_cache.items():
                    entries[key.file_key] = [key.runner_class_name, self.encode(tactic)]
        return entries

    def verify_entries(self, stage, entries):
        check = compare_tactics(self.reference, entries)
        check["entries_sha256"] = entries_sha256(entries)
        self.report["checks"][stage] = check
        (self.out / f"{stage}_tactics.json").write_text(json.dumps(entries, indent=2) + "\n")
        self.write_report()
        if check["missing"] or check["changed"]:
            raise RuntimeError(f"imported FlashInfer tactics disagree at {stage}; see autotune/verification.json")

    def observe(self, key, runner_name, tactic, hit):
        if hit:
            value = [runner_name, tactic]
            event = self.observed.setdefault((self.phase, key, json.dumps(value, sort_keys=True)),
                                            {"phase": self.phase, "key": key, "actual": value,
                                             "expected": self.reference.get(key), "calls": 0})
        else:
            event = self.misses.setdefault((self.phase, key),
                                          {"phase": self.phase, "key": key, "calls": 0})
        event["calls"] += 1

    def install(self, *, original_warmup=None):
        # Set before vLLM imports/cache resolution. Only JSON is copied here;
        # its torch.compile and FlashInfer JIT caches stay isolated and fresh.
        os.environ["VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"] = str(self.cache_root)
        from flashinfer.autotuner import AutoTuner
        from flashinfer.autotuner.autotuner import _tactic_to_json
        warmup = importlib.import_module("vllm.model_executor.warmup.kernel_warmup")
        from vllm.distributed.parallel_state import get_world_group
        self.encode = _tactic_to_json
        files = {"flashinfer_autotuner": inspect.getsourcefile(AutoTuner),
                 "vllm_kernel_warmup": getattr(warmup, "__file__", None)}
        self.report["runtime_source_sha256"] = {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in files.items() if path}
        self.write_report()
        original = original_warmup or warmup.flashinfer_autotune
        owner = self

        @functools.wraps(original)
        def run(runner, *args, **kwargs):
            if owner.tuner is not None:
                raise RuntimeError("autotune import expects one single-GPU warmup")
            world = get_world_group()
            parallel = runner.vllm_config.parallel_config
            if world.world_size != 1 or parallel.data_parallel_size != 1:
                raise RuntimeError("autotune import supports only one GPU/rank")
            path = warmup.resolve_flashinfer_autotune_file(runner)
            path = path.with_name(f"{path.stem}_dp{parallel.data_parallel_rank}_rank{world.rank_in_group}{path.suffix}")
            if not path.resolve().is_relative_to(owner.cache_root.resolve()):
                raise RuntimeError("vLLM resolved autotune cache outside this run's isolated import directory")
            # This is the actual vLLM-resolved rank cache path, in the new run's
            # isolated cache mount. Never overwrite another file or copy graphs.
            with path.open("xb") as f:
                f.write(owner.source_bytes)
            owner.report["seeded_cache_path"] = str(path)
            owner.tuner = tuner = AutoTuner.get()
            if tuner._winner_cache() is not tuner.profiling_cache:
                raise RuntimeError("managed/v2 FlashInfer winner caches are outside this probe")
            load, search = tuner.load_configs, tuner.search_cache

            def observed_load(filename):
                loaded = load(filename)
                owner.report["load_calls"].append({"path": str(filename), "returned": bool(loaded)})
                owner.write_report()
                if Path(filename) != path or loaded is not True:
                    raise RuntimeError("FlashInfer did not successfully load the imported rank cache")
                owner.verify_entries("loaded", owner._effective_entries(False))
                return loaded

            def observed_search(custom_op, runners, input_shapes, tuning_config, inputs=None):
                result = search(custom_op, runners, input_shapes, tuning_config, inputs)
                hit, runner_id, tactic, _ = result
                selected = [(runner_id, runners[runner_id])] if hit else list(enumerate(runners))
                for _, candidate in selected:
                    key = AutoTuner._get_cache_key(
                        custom_op, candidate, input_shapes, tuning_config,
                        candidate.get_cache_key_extras(inputs) if inputs is not None else ())
                    owner.observe(key.file_key, type(candidate).__name__,
                                  owner.encode(tactic) if hit else None, hit)
                return result  # Keep the original selected runner/tactic unchanged.

            tuner.load_configs = observed_load
            if owner.observe_search:
                tuner.search_cache = observed_search
            try:
                result = original(runner, *args, **kwargs)
                if not owner.report["load_calls"]:
                    raise RuntimeError("vLLM did not attempt to load the imported autotune cache")
                owner.verify_entries("after_warmup", owner._effective_entries(True))
                owner.report["status"] = "loaded_and_warmup_verified"
                owner.phase = "graph_capture_or_inference"
                owner.write_report()
                return result
            finally:
                tuner.load_configs = load
                # Keep the read-only search observer through graph capture and
                # generation. Its audit is not a latency baseline.

        warmup.flashinfer_autotune = run

    def finalize(self, stage="after_generation"):
        error = None
        events = list(self.observed.values())
        changed = [e for e in events if e["expected"] is not None and e["expected"] != e["actual"]]
        uncovered = [e for e in events if e["expected"] is None]
        misses = list(self.misses.values())
        self.report.update(observed_cache_hits=events, cache_misses=misses,
                           observed_reference_mismatches=changed,
                           uncovered_cache_hits=uncovered,
                           all_observed_lookups_covered=bool(events) and not uncovered and not misses)
        try:
            if (not self.report["load_calls"] or self.tuner is None
                    or not all(call["returned"] for call in self.report["load_calls"])
                    or "after_warmup" not in self.report["checks"]):
                raise RuntimeError("autotune import was not loaded; matched tactics are unverified")
            if self.tuner._winner_cache() is not self.tuner.profiling_cache:
                raise RuntimeError("active FI winner cache changed to an unsupported partition")
            effective = self._effective_entries(True)
            self.verify_entries(stage, effective)
            # Export a complete FI-loadable snapshot without save_configs side
            # effects, while preserving the separate normalized-entry hash.
            snapshot = {"_metadata": self.document["_metadata"], **effective}
            (self.out / f"{stage}_effective.json").write_text(json.dumps(snapshot, indent=2) + "\n")
            if changed:
                raise RuntimeError("observed FlashInfer choices differ from imported tactics")
            if self.require_coverage and not self.report["all_observed_lookups_covered"]:
                raise RuntimeError("FlashInfer lookups were not fully covered by imported tactics")
            self.report["status"] = ("reference_tactics_verified_without_lookup_observation"
                                     if not self.observe_search else
                                     "reference_and_observed_tactics_verified"
                                     if self.report["all_observed_lookups_covered"] else
                                     "reference_verified_with_uncovered_lookups")
            self.report["last_verified_stage"] = stage
        except BaseException as ex:
            error = ex
            self.report.update(status="verification_failed", error=f"{type(ex).__name__}: {ex}")
        finally:
            # FI choices alone cannot establish identical generated kernels.
            # Preserve available compiler choices for the paired-run audit.
            cache_root = Path(os.environ.get("VLLM_CACHE_ROOT", str(self.out.parent / "cache/vllm")))
            try:
                audit = compiler_config_audit(cache_root)
                (self.out / "compiler_config_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
                self.report["selected_inductor_config_count"] = len(audit["files"])
            except (OSError, ValueError) as ex:
                self.report["compiler_config_audit_error"] = str(ex)
            self.write_report()
        if error is not None:
            raise error
        return self.report
