"""Snapshot exact source files used by the isolated Qwen experiment runner."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selected(variant, fi_control=False):
    paths = [
        'scripts/serve.sh', 'scripts/cudagraph_sizes.txt',
        'bench/qwen_experiment.sh', 'bench/serving_matrix.py', 'bench/source_manifest.py',
        'bench/agentic_bench.py', 'eval/repeated_nll.py', 'eval/long_eval.py',
        'eval/ckpt_dist.py', 'kernels/moe-nvfp4/verify_logprobs.py',
    ]
    for relative in (
        'v1/attention/backends/flashinfer.py', 'v1/attention/backends/sm120_fp8_fmha_glue.py',
        'model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py',
        'model_executor/layers/mamba/ops/fused_conv_post_conv.py',
    ):
        if variant.endswith('_gdn') and '/gdn/' in relative:
            continue
        paths.append('build/vllm/' + relative)
    if variant.endswith('_gdn'):
        paths.extend(('kernels/gdn-norm-fp8/integration_config.py', 'bench/verify_gdn_startup.py'))
        paths.extend('kernels/gdn-norm-fp8/overlay/vllm/model_executor/layers/mamba/gdn/' + name
                     for name in ('qwen_gdn_linear_attn.py', 'gdn_norm_fp8_fusion.py',
                                  'gdn_norm_fp8_guards.py', 'gdn_norm_fp8_kernel.py', 'gdn_norm_fp8_quant_out.py'))
    moe_variant = variant.removesuffix('_gdn')
    if moe_variant != 'next':
        paths += [
            f'kernels/moe-nvfp4/{moe_variant}/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py',
            'kernels/moe-nvfp4/patched/vllm/model_executor/layers/fused_moe/oracle/unquantized.py',
        ]
    if fi_control:
        paths.extend('kernels/decode-route-capture/' + name for name in (
            'qwen_fi_control_plugin.py', 'autotune_import.py',
            'references/next_cpu_audit_autotune.json',
            'references/next_cpu_audit_autotune.provenance.json',
            'qwen_fi_control-0.1.dist-info/entry_points.txt',
            'qwen_fi_control-0.1.dist-info/METADATA'))
    return paths


def snapshot(root, output, variant, fi_control=False):
    files = selected(variant, fi_control)
    if (output / 'source_manifest.json').exists():
        raise ValueError('source manifest already exists')
    contents = {relative: (root / relative).read_bytes() for relative in files}
    manifest = {'schema_version': 1, 'variant': variant, 'fi_control': fi_control, 'files': {}}
    for relative, data in contents.items():
        path = output / 'source_files' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest['files'][relative] = hashlib.sha256(data).hexdigest()
    (output / 'source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def verify(root, output):
    manifest = json.loads((output / 'source_manifest.json').read_text())
    changed = []
    for relative, expected in manifest['files'].items():
        for label, path in [('live', root / relative), ('snapshot', output / 'source_files' / relative)]:
            if not path.is_file() or sha(path) != expected:
                changed.append(f'{label}:{relative}')
    if changed:
        raise ValueError('experiment source changed: ' + ', '.join(changed))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('snapshot', 'verify'))
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--variant', choices=('next', 'shared', 'shared_exact', 'scalar', 'shared_exact_gdn', 'scalar_gdn'))
    parser.add_argument('--fi-control', action='store_true')
    args = parser.parse_args()
    if args.action == 'snapshot' and args.variant is None:
        parser.error('snapshot needs --variant')
    result = (snapshot(args.root, args.output, args.variant, args.fi_control) if args.action == 'snapshot'
              else verify(args.root, args.output))
    print(json.dumps({'status': 'passed', 'action': args.action, 'files': len(result['files']),
                      'variant': result['variant']}))


if __name__ == '__main__':
    main()
