"""Process-local RN header experiment; installed RU source stays untouched."""
import hashlib,json
from pathlib import Path

def install(mode):
    from humming.jit.compiler import Compiler
    root=Path(__file__).resolve().parent/'rounding'
    manifest=json.loads((root/'manifest.json').read_text())
    original=Path(Compiler.humming_include_dir())/'humming/kernel/process_input/quantization.cuh'
    assert hashlib.sha256(original.read_bytes()).hexdigest()==manifest['original_header_sha256']
    if mode=='ru':return dict(rounding=mode,header_sha256=manifest['original_header_sha256'])
    assert mode=='rn'
    private=root/'include_rn'
    for name,sha in manifest['all_headers_sha256'].items():
        assert hashlib.sha256((private/name).read_bytes()).hexdigest()==sha
    Compiler.humming_include_dir=staticmethod(lambda:str(private))
    # Compiler.compile includes the actual header paths/mtime set in its key;
    # launcher additionally isolates HUMMING_CACHE_DIR by mode and unique tag.
    return dict(rounding=mode,header_sha256=manifest['rn_header_sha256'],include_dir=str(private))
