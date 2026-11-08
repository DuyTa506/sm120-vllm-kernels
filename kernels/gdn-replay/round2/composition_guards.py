"""CPU-only fail-closed composition and active-source checks."""
import hashlib
import json
from pathlib import Path


def replace_once(text, before, after):
    count = text.count(before)
    if count != 1:
        raise ValueError(f'expected exactly one replacement, found {count}: {before[:100]!r}')
    return text.replace(before, after, 1)


def verify_runtime_sources(manifest_path, expected_manifest_sha, package_root):
    manifest_path, package_root = Path(manifest_path), Path(package_root)
    observed_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if observed_sha != expected_manifest_sha:
        raise ValueError('composition manifest changed or environment identity is wrong')
    manifest = json.loads(manifest_path.read_text())
    actual = {}
    for name, expected in manifest['generated_sha256'].items():
        if not name.startswith('vllm/') or name.endswith('.orig'):
            continue
        path = package_root / name.removeprefix('vllm/')
        value = hashlib.sha256(path.read_bytes()).hexdigest()
        if value != expected:
            raise ValueError(f'active source differs from composition: {name}')
        actual[name] = value
    required = {'vllm/envs.py', 'vllm/v1/worker/gpu/model_states/mamba_hybrid.py',
                'vllm/model_executor/layers/mamba/gdn/gdn_state_replay.py',
                'vllm/model_executor/layers/mamba/gdn/gdn_replay.cu',
                'vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py'}
    if not required <= actual.keys():
        raise ValueError('composition manifest is missing required replay files')
    return actual
