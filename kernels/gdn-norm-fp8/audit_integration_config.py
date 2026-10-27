#!/usr/bin/env python3
"""CPU-only exact installed split/pass completion audit; no vLLM/Torch import."""
import hashlib
import json
from pathlib import Path
import sys

from integration_config import OP, add_split
from test_integration_guards import resolve_installed_split_defaults


def main():
    root=Path(__file__).resolve().parent
    source_root=Path('/usr/local/lib/python3.12/dist-packages/vllm/config')
    sources={name:(source_root/name).read_text() for name in ('compilation.py','vllm.py')}
    old_path=root.parent.parent/'results/qwen_sequential/gdn_integration_1/compilation_config.json'
    old=json.loads(old_path.read_text())
    candidate,standard=add_split({},sources['compilation.py'])
    resolve=lambda c:resolve_installed_split_defaults(c,sources['compilation.py'],sources['vllm.py'])
    baseline=resolve({});before=resolve(old);after=resolve(candidate)
    assert len(baseline['splitting_ops'])==19 and len(after['splitting_ops'])==20
    assert [x for x in after['splitting_ops'] if x!=OP]==baseline['splitting_ops']
    assert after['pass_config']==baseline['pass_config']
    assert before['pass_config']['fuse_rope_kvcache_cat_mla'] is True
    assert after['pass_config']['fuse_rope_kvcache_cat_mla'] is False
    report=dict(cuda_imported='torch' in sys.modules,
                source_sha256={name:hashlib.sha256(value.encode()).hexdigest() for name,value in sources.items()},
                baseline=baseline,first_failed_candidate=before,corrected_candidate=after,
                emitted_candidate=candidate,baseline_split_count=19,candidate_split_count=20,
                exact_methods=['CompilationConfig.set_splitting_ops_for_v1',
                    'CompilationConfig.splitting_ops_contain_kv_cache_update','enable_rope_kvcache_mla_fusion'],
                scope='Executes the installed relevant default-completion methods with the observed mode3/no-partition/no-attn-quant configuration; no full VllmConfig/platform construction, GPU or server launch')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
