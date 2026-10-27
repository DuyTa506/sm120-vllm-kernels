"""Strict offline identity checks shared by Qwen serving comparators."""
import ast
import csv
import hashlib
import io
import importlib.util
import json
import math
from pathlib import Path
import re

PKG = '/usr/local/lib/python3.12/dist-packages/vllm/'
EXPERT = 'model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py'
ORACLE = 'model_executor/layers/fused_moe/oracle/unquantized.py'
GDN = 'model_executor/layers/mamba/gdn/'
GDN_ROOT = 'kernels/gdn-norm-fp8/overlay/vllm/' + GDN
GDN_FILES = ('qwen_gdn_linear_attn.py', 'gdn_norm_fp8_fusion.py', 'gdn_norm_fp8_guards.py',
             'gdn_norm_fp8_kernel.py', 'gdn_norm_fp8_quant_out.py')
GDN_OP = 'vllm::gdn_attention_core_norm_fp8_packed_experimental'
# Installed vLLM 0.32 CompilationConfig._attention_ops, source SHA256
# d8a8086166e3f344b75d98d8350a77592f7b6c145319b3a9f0b9c8cb36bc39cb.
STANDARD_SPLITS = ['vllm::' + x for x in (
    'unified_attention_with_output', 'unified_mla_attention_with_output', 'mamba_mixer2',
    'mamba_mixer', 'short_conv', 'qwen4_exp_ple_short_conv', 'qwen4_exp_qsa_with_output',
    'qwen4_exp_qsa_prepare_with_output', 'linear_attention', 'qwen_gdn_attention_core',
    'qwen_gdn_attention_core_fused_norm_packed', 'gdn_attention_core_xpu',
    'olmo_hybrid_gdn_full_forward', 'sparse_attn_indexer', 'rocm_aiter_sparse_attn_indexer',
    'deepseek_v4_attention', 'hpc_rope_norm_forward')]
AUTO_SPLITS = ['vllm::unified_kv_cache_update', 'vllm::unified_mla_kv_cache_update']
VARIANTS = ('next', 'shared', 'shared_exact', 'scalar', 'shared_exact_gdn', 'scalar_gdn')
VARIANT_ENV = {'VLLM_CACHE_ROOT', 'VLLM_B12X_FIX_SCALES', 'VLLM_B12X_SHARED_WORKSPACE',
               'VLLM_B12X_SCALAR_INPUT_SCALE', 'VLLM_GDN_NORM_FP8_FUSION'}

FI_ROOT = 'kernels/decode-route-capture/'
FI_FILES = ('qwen_fi_control_plugin.py', 'autotune_import.py',
            'references/next_cpu_audit_autotune.json', 'references/next_cpu_audit_autotune.provenance.json',
            'qwen_fi_control-0.1.dist-info/entry_points.txt', 'qwen_fi_control-0.1.dist-info/METADATA')
FI_SHA = '7b5146f9b3c1e7cc09586c2087988ea8aaf9da1d86da036539a32dec4b335ec3'
FI_ENV = {'PYTHONPATH':'/opt/qwen-fi-control',
          'QWEN_FI_IMPORT_JSON':'/opt/qwen-fi-control/references/next_cpu_audit_autotune.json',
          'QWEN_FI_IMPORT_SHA256':FI_SHA, 'QWEN_FI_IMPORT_OUT':'/bench-results/fi_control',
          'VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR':'/bench-results/fi_control/cache/autotune_import'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()



MEMORY_FIELDS = ('kv_available_gib', 'kv_capacity_tokens', 'concurrency_request_tokens',
                 'maximum_concurrency', 'cuda_graph_actual_gib', 'cuda_graph_estimated_gib')


def parse_startup_memory(startup):
    """Read observed startup capacity; never use profiling capture as final pool size."""
    lines = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', startup).splitlines()
    result = {key: None for key in MEMORY_FIELDS}
    evidence = {}

    def records(marker):
        return [line for line in lines if marker in line]

    def unique(marker, pattern):
        found = records(marker)
        require(sum(line.count(marker) for line in found) <= 1, 'ambiguous startup memory record: ' + marker)
        if not found:
            return None
        match = re.search(pattern, found[0])
        require(match is not None, 'malformed startup memory record: ' + marker)
        evidence[marker] = found[0]
        return match.groups()

    def decimal(value, label, positive=False):
        require(re.fullmatch(r'(?:0|[1-9]\d*)(?:\.\d+)?', value) is not None,
                'invalid startup memory number: ' + label)
        number = float(value)
        require(math.isfinite(number) and (number > 0 if positive else number >= 0),
                'invalid startup memory number: ' + label)
        return number

    def integer(value, label):
        require(re.fullmatch(r'(?:[1-9]\d*|[1-9]\d{0,2}(?:,\d{3})+)', value) is not None,
                'invalid startup memory integer: ' + label)
        return int(value.replace(',', ''))

    available = unique('Available KV cache memory:', r'Available KV cache memory:\s+(\S+)\s+GiB\s*$')
    if available:
        result['kv_available_gib'] = decimal(available[0], 'available KV GiB', positive=True)
    capacity = unique('GPU KV cache size:', r'GPU KV cache size:\s+(\S+)\s+tokens(?:,|\s*$)')
    if capacity:
        result['kv_capacity_tokens'] = integer(capacity[0], 'KV capacity tokens')
    concurrency = unique('Maximum concurrency for',
        r'Maximum concurrency for\s+(\S+)\s+tokens per request:\s+(\S+)x\s*$')
    if concurrency:
        result['concurrency_request_tokens'] = integer(concurrency[0], 'concurrency request tokens')
        result['maximum_concurrency'] = decimal(concurrency[1], 'maximum concurrency', positive=True)
        if capacity:
            precision = len(concurrency[1].partition('.')[2])
            tolerance = .5 * 10 ** -precision + 1e-9
            expected = result['kv_capacity_tokens'] / result['concurrency_request_tokens']
            require(abs(result['maximum_concurrency'] - expected) <= tolerance,
                    'KV capacity and rounded maximum concurrency disagree')
    graph = unique('CUDA graph pool memory:',
        r'CUDA graph pool memory:\s+(\S+)\s+GiB \(actual\),\s+(\S+)\s+GiB \(estimated\)(?:,.*)?\s*$')
    if graph:
        result['cuda_graph_actual_gib'] = decimal(graph[0], 'final actual CUDA graph GiB')
        result['cuda_graph_estimated_gib'] = decimal(graph[1], 'estimated CUDA graph GiB')
    captures = []
    for line in records('Graph capturing finished in'):
        match = re.search(r'Graph capturing finished in\s+(\S+)\s+secs, took\s+(\S+)\s+GiB\s*$', line)
        require(match is not None, 'malformed graph capture completion record')
        decimal(match[1], 'graph capture seconds')
        captures.append(decimal(match[2], 'graph capture GiB'))
    if graph and captures:
        require(math.isclose(captures[-1], result['cuda_graph_actual_gib'], rel_tol=0, abs_tol=.0050001),
                'last graph capture and final actual graph pool disagree')
    missing = [key for key in MEMORY_FIELDS if result[key] is None]
    return {**result, 'status': 'missing' if len(missing) == len(MEMORY_FIELDS) else 'partial' if missing else 'complete',
            'missing_fields': missing, 'missing_reason': 'not recorded in this startup log' if missing else None,
            'graph_capture_events_gib': captures, 'source': 'startup.log', 'startup_sha256': sha(startup.encode()),
            'evidence_lines': evidence,
            'scope': 'Logged startup allocation/capacity context; not a peak-memory benchmark or causal attribution.'}


def paired_startup_memory(base, candidate):
    return {'baseline': base, 'candidate': candidate,
            'delta_candidate_minus_baseline': {key: candidate[key] - base[key]
                if candidate[key] is not None and base[key] is not None else None for key in MEMORY_FIELDS},
            'identity_equality_required': False,
            'interpretation': 'Startup capacity and graph allocation can legitimately differ. These rounded log values are context only; '
                              'they do not establish a memory saving caused by an overlay. Configuration, FI control and prior history still need their separate compatibility checks.'}


def format_startup_memory(context):
    a, b = context['baseline'], context['candidate']
    def pair(key, integer=False):
        return ' / '.join('not recorded' if run[key] is None else
                          f'{run[key]:,}' if integer else f'{run[key]:.2f}' for run in (a, b))
    return (f"Startup memory (baseline / candidate): available KV {pair('kv_available_gib')} GiB; "
            f"KV capacity {pair('kv_capacity_tokens', True)} tokens; maximum concurrency {pair('maximum_concurrency')}x "
            f"at {pair('concurrency_request_tokens', True)} tokens/request; final CUDA graph pool "
            f"actual {pair('cuda_graph_actual_gib')} GiB, estimated {pair('cuda_graph_estimated_gib')} GiB. "
            f"Extraction: {a['status']} / {b['status']}. " + context['interpretation'])


def option(argv, key):
    require(argv.count(key) == 1 and argv.index(key) + 1 < len(argv), f'missing/duplicate {key}')
    return argv[argv.index(key) + 1]


def parent(variant):
    require(variant in VARIANTS, f'unknown overlay variant: {variant}')
    return variant.removesuffix('_gdn')


def source_extras(variant):
    moe = parent(variant)
    paths = set() if moe == 'next' else {
        f'kernels/moe-nvfp4/{moe}/vllm/{EXPERT}', f'kernels/moe-nvfp4/patched/vllm/{ORACLE}'}
    if variant.endswith('_gdn'):
        paths |= {GDN_ROOT + name for name in GDN_FILES}
        paths |= {'kernels/gdn-norm-fp8/integration_config.py', 'bench/verify_gdn_startup.py'}
    return paths


def expected_mounts(variant):
    paths = ('v1/attention/backends/flashinfer.py', 'v1/attention/backends/sm120_fp8_fmha_glue.py',
             'model_executor/layers/mamba/ops/fused_conv_post_conv.py')
    mounts = {PKG + p: 'build/vllm/' + p for p in paths}
    if variant.endswith('_gdn'):
        mounts.update({PKG + GDN + name: GDN_ROOT + name for name in GDN_FILES})
    else:
        mounts[PKG + GDN + GDN_FILES[0]] = 'build/vllm/' + GDN + GDN_FILES[0]
    if parent(variant) != 'next':
        mounts[PKG + EXPERT] = f'kernels/moe-nvfp4/{parent(variant)}/vllm/{EXPERT}'
        mounts[PKG + ORACLE] = f'kernels/moe-nvfp4/patched/vllm/{ORACLE}'
    return mounts


def verify_fi(root, text, read, sources):
    required = {FI_ROOT + name for name in FI_FILES}
    require(required <= sources['files'].keys(), 'FI control sources missing from snapshot')
    local = Path(__file__).resolve().parents[1] / FI_ROOT
    # Execute only the reviewed local verifier, after matching its snapshot;
    # never execute Python supplied by an artifact directory.
    for name in ('qwen_fi_control_plugin.py', 'autotune_import.py'):
        require(sources['files'][FI_ROOT + name] == sha((local/name).read_bytes()),
                'unsupported FI verifier source version: ' + name)
    spec = importlib.util.spec_from_file_location('_paired_fi_verifier', local/'qwen_fi_control_plugin.py')
    plugin = importlib.util.module_from_spec(spec); spec.loader.exec_module(plugin)
    saved_source = read('serving/fi_control/autotune/source.json')
    require(sha(text('serving/fi_control/autotune/source.json').encode()) == FI_SHA
            and sources['files'][FI_ROOT + 'references/next_cpu_audit_autotune.json'] == FI_SHA,
            'FI immutable reference SHA mismatch')
    helper = plugin._helpers()
    reference = helper.tactic_entries(saved_source)
    entries_sha = helper.entries_sha256(reference)
    provenance = read('source_files/' + FI_ROOT + 'references/next_cpu_audit_autotune.provenance.json')
    require(provenance['reference_sha256'] == FI_SHA and provenance['reference_entries_sha256'] == entries_sha
            and provenance['entries'] == len(reference), 'FI reference provenance mismatch')
    verification = read('serving/fi_control/autotune/verification.json')
    require(verification.get('status') == 'reference_tactics_verified_without_lookup_observation'
            and verification.get('compiled_graph_cache_imported') is False
            and verification.get('lookup_observation_enabled') is False
            and verification.get('serving_lookup_observer') is False,
            'FI control is unverified or has unsupported runtime observation/import')
    require(verification.get('reference_entries_sha256') == entries_sha, 'FI report reference is not the saved source')
    require(verification.get('import_mode') == 'fixed_reference_replay_v1'
            and verification.get('preload',{}).get('returned') is True
            and verification.get('preload',{}).get('boundary') == 'before_worker_load_model'
            and verification.get('server_startup_boundary') == 'after_compile_or_warm_up_model',
            'FI fixed replay/preload/after-capture boundary evidence missing')
    calls = verification.get('load_calls')
    require(calls and all(x.get('returned') is True for x in calls), 'FI load acceptance evidence missing/failed')
    for stage in ('preloaded', 'loaded', 'after_warmup', 'server_startup', 'server_shutdown'):
        check = verification.get('checks', {}).get(stage, {})
        require(check.get('reference_entries') == check.get('actual_entries') == len(reference)
                and check.get('entries_sha256') == entries_sha
                and all(check.get(k) == [] for k in ('missing', 'extra', 'changed')),
                'FI effective tactic check failed/missing: ' + stage)
        require(read('serving/fi_control/autotune/' + stage + '_tactics.json') == reference,
                'FI stage tactic artifact differs from reference: ' + stage)
    verified = {}
    for stage, shutdown in (('startup',False),('shutdown',True)):
        # Capture every consumed artifact's hash through the caller.
        document = read('serving/fi_control/autotune/server_' + stage + '_effective.json')
        require(document['_metadata'] == saved_source['_metadata'], 'FI effective metadata mismatch')
        actual = plugin.verify_artifacts(root/'serving/fi_control', FI_SHA, require_shutdown=shutdown)
        recorded = read('fi_' + stage + '_verify.json')
        require(recorded == actual, 'FI saved verification differs from artifact recomputation: ' + stage)
        require(actual['effective_entries_sha256'] == entries_sha and actual['entries'] == len(reference),
                'FI effective choices differ from saved source')
        verified[stage] = actual
    runtime_hashes=verification.get('runtime_source_sha256', {})
    require(set(runtime_hashes)=={'flashinfer_autotuner','vllm_kernel_warmup'}
            and all(isinstance(v,str) and re.fullmatch(r'[0-9a-f]{64}',v) for v in runtime_hashes.values()),
            'FI installed runtime source hashes missing/invalid')
    compiler = read('serving/fi_control/autotune/compiler_config_audit.json')
    require(compiler['files'] and not verification.get('compiler_config_audit_error'), 'FI compiler config audit missing/failed')
    require(len(compiler['files']) == verification.get('selected_inductor_config_count'), 'FI compiler audit count mismatch')
    for entry in compiler['files']:
        digest = sha(json.dumps(entry['selected'],sort_keys=True,separators=(',',':')).encode())
        require(entry['selected_sha256'] == digest, 'FI selected Inductor config hash mismatch')
    return {'reference_sha256':FI_SHA, 'reference_entries_sha256':entries_sha, 'verified':verified,
            'import_mode':verification['import_mode'], 'preload':verification['preload'],
            'server_startup_boundary':verification['server_startup_boundary'],
            'runtime_source_sha256':verification.get('runtime_source_sha256', {}),
            'compiler_config_audit':compiler, 'inductor_pinned':False, 'serving_lookup_observer':False}


def load(root, text, read, *, allow_legacy_sources=False):
    config = dict(line.split('=', 1) for line in text('config.txt').splitlines())
    require(config['tag'] == root.name, 'run tag/root mismatch')
    variant = config['variant']; moe = parent(variant); gdn = variant.endswith('_gdn')
    require(config.get('fi_control', '0') in ('0','1'), 'unknown FI control mode')
    fi_control = config.get('fi_control', '0') == '1'
    require(config.get('fi_source_sha256') == (FI_SHA if fi_control else None), 'unexpected FI source identity')
    image = text('image.txt').strip()
    commit, diff_hash = text('commit.txt').strip(), sha(text('uncommitted.diff').encode())
    containers = read('container.json')
    require(isinstance(containers, list) and len(containers) == 1, 'invalid container capture')
    container = containers[0]
    require(container['Name'] == '/qwen-exp-' + config['tag'], 'container identity mismatch')
    require(container['Image'] == image, 'actual container image differs from captured image')
    raw_command = container['Config']['Cmd']; command = raw_command[:]
    if fi_control:
        require(option(command, '--shutdown-timeout') == '30'
                and not any(arg.startswith('--shutdown-timeout=') for arg in command),
                'FI control requires exactly one --shutdown-timeout 30 for shutdown evidence')
    if moe != 'next':
        require(option(command, '--moe-backend') == 'flashinfer_b12x', 'candidate backend mismatch')
        pos = command.index('--moe-backend'); del command[pos:pos+2]
    else:
        require('--moe-backend' not in command, 'baseline unexpectedly overrides the MoE backend')
    expected_compilation = {'splitting_ops': STANDARD_SPLITS + AUTO_SPLITS + [GDN_OP],
                            'pass_config':{'fuse_rope_kvcache_cat_mla':False}}
    startup = text('startup.log')
    engine_configs = re.findall(r'Initializing a V1 LLM engine \(v[^)]*\) with config: ([^\n]+)', startup)
    require(len(engine_configs) == 1, 'missing/ambiguous resolved engine config')
    engine_raw = engine_configs[0]
    splits = re.findall(r"'splitting_ops': (\[[^\n]*?\])", engine_raw)
    require(len(splits) == 1, 'missing/ambiguous resolved splitting ops')
    effective_splits = ast.literal_eval(splits[0])
    expected_splits = STANDARD_SPLITS + AUTO_SPLITS + ([GDN_OP] if gdn else [])
    require(effective_splits == expected_splits, 'effective startup split ops differ from required config')
    engine = engine_raw.replace(splits[0], repr(STANDARD_SPLITS + AUTO_SPLITS), 1)
    if moe != 'next':
        require(engine.count("moe_backend='flashinfer_b12x'") == 1, 'resolved MoE backend mismatch')
        engine = engine.replace("moe_backend='flashinfer_b12x'", "moe_backend='auto'", 1)
    else:
        require(engine.count("moe_backend='auto'") == 1, 'resolved baseline MoE backend mismatch')
    compilation = None; readiness = None
    if gdn:
        compilation = read('compilation_config.json')
        require(compilation == expected_compilation, 'unexpected GDN compilation config fields/split ops')
        require(json.loads(option(command, '--compilation-config')) == compilation, 'actual GDN compilation config mismatch')
        pos = command.index('--compilation-config'); del command[pos:pos+2]
        require("'fuse_rope_kvcache_cat_mla': False" in engine_raw, 'GDN changed resolved rope/KV fusion policy')
        expected = [i for i in range(40) if (i+1) % 4]
        ready_lines = [line for line in startup.splitlines() if 'GDN norm-FP8 ready for ' in line]
        matches = [re.search(r'layers\.(\d+)\.linear_attn', line) for line in ready_lines]
        require(all(matches), 'malformed GDN readiness log')
        layers = sorted(int(m.group(1)) for m in matches)
        disabled = [line for line in startup.splitlines() if 'GDN norm-FP8 disabled' in line]
        readiness = read('gdn_readiness.json')
        require(layers == expected and not disabled, 'missing/duplicate/disabled GDN layers at startup')
        require(readiness['ready_layers'] == layers and readiness['expected_layers'] == expected
                and readiness['disabled'] == disabled, 'GDN readiness artifact/log mismatch')
    else:
        require('--compilation-config' not in command, 'unexpected compilation override on non-GDN run')
    speculative = json.loads(option(command, '--speculative-config'))
    require(speculative.get('method') == 'mtp' and speculative.get('num_speculative_tokens', 0) > 0,
            'comparison requires MTP enabled')
    env = dict(value.split('=', 1) for value in container['Config']['Env'])
    require(len(env) == len(container['Config']['Env']), 'duplicate environment keys')
    if fi_control:
        require(all(env.get(k) == v for k,v in FI_ENV.items()), 'FI environment does not match fixed control contract')
    else:
        require(not (set(FI_ENV)-{'PYTHONPATH'}) & env.keys(), 'FI environment present on uncontrolled run')
    expected_flags = {'VLLM_B12X_FIX_SCALES': moe != 'next', 'VLLM_B12X_SHARED_WORKSPACE': moe != 'next',
                      'VLLM_B12X_SCALAR_INPUT_SCALE': moe == 'scalar', 'VLLM_GDN_NORM_FP8_FUSION': gdn}
    for key, enabled in expected_flags.items():
        require(env.get(key, '0') == str(int(enabled)), f'{key} does not match variant')
    require(env.get('VLLM_CACHE_ROOT') == '/root/.cache/vllm/qwen-sequential-' + variant,
            'unexpected isolated cache root')
    gpu_rows = list(csv.DictReader(io.StringIO(text('gpu_before.csv')), skipinitialspace=True))
    require(len(gpu_rows) == 1, 'expected one GPU identity')
    gpu = {key: gpu_rows[0][key].strip() for key in
           ('name', 'uuid', 'driver_version', 'memory.total [MiB]', 'power.limit [W]')}
    legacy_sources = not (root/'source_manifest.json').is_file()
    if legacy_sources:
        require(allow_legacy_sources and not gdn and not fi_control,
                'strict comparison requires a runtime source manifest')
        sources = {'schema_version':None, 'variant':variant, 'files':{}}
        fi_evidence = None
    else:
        sources = read('source_manifest.json')
        require(sources['schema_version'] == 1 and sources['variant'] == variant, 'source manifest identity mismatch')
        for relative, digest in sources['files'].items():
            path = Path(relative)
            require(not path.is_absolute() and '..' not in path.parts, 'unsafe source path')
            require(sha(text('source_files/' + relative).encode()) == digest, 'source snapshot hash mismatch: ' + relative)
        verification = read('source_verify.log')
        require(verification.get('status') == 'passed' and verification.get('action') == 'verify'
                and verification.get('variant') == variant and verification.get('files') == len(sources['files']),
                'source verification failed/missing')
        require(sources.get('fi_control', False) is fi_control, 'FI source manifest mode mismatch')
        fi_evidence = verify_fi(root, text, read, sources) if fi_control else None
        required = source_extras(variant) | set(expected_mounts(variant).values()) | {
            'scripts/serve.sh', 'scripts/cudagraph_sizes.txt', 'bench/qwen_experiment.sh', 'bench/serving_matrix.py', 'bench/source_manifest.py'}
        require(required <= sources['files'].keys(), 'required runtime source not captured: ' + str(sorted(required - sources['files'].keys())))
    mounts = container['Mounts']; destinations = [m['Destination'] for m in mounts]
    require(len(set(destinations)) == len(destinations), 'duplicate mount destination')
    expected = expected_mounts(variant)
    external = {'/bench-results', '/root/.cache/vllm', '/root/.cache/flashinfer', '/root/.cache/huggingface'}
    if fi_control:
        external.add('/opt/qwen-fi-control')
    require(set(destinations) == set(expected) | external, 'unexpected/missing mount destinations: ' +
            str(sorted(set(destinations) ^ (set(expected) | external))))
    common_mounts = {}
    for mount in mounts:
        dest = mount['Destination']
        require(mount['Type'] == 'bind', 'runtime mount is not a bind')
        if dest in expected:
            require(mount['RW'] is False and mount['Mode'] == 'ro', 'runtime Python source was mounted writable')
            require(mount['Source'].endswith('/' + expected[dest]), 'wrong source mounted at ' + dest)
        else:
            normalized = dict(mount)
            if dest == '/opt/qwen-fi-control':
                require(mount['Source'].endswith('/kernels/decode-route-capture') and mount['RW'] is False
                        and mount['Mode'] == 'ro', 'FI control mount identity/permissions mismatch')
            if dest == '/bench-results':
                require(mount['Source'].endswith('/results/qwen_sequential/' + root.name + '/serving'), 'wrong result directory mount')
                normalized['Source'] = 'RUN_RESULTS'
            common_mounts[dest] = normalized
    return {'name': root.name, 'config': config, 'variant_parent': moe, 'gdn_enabled': gdn,
            'image': image, 'commit': commit, 'uncommitted_diff_sha256': diff_hash, 'gpu': gpu,
            'server_command': command, 'server_command_raw': raw_command,
            'resolved_engine_config':engine, 'resolved_engine_config_raw':engine_raw,
            'startup_memory':parse_startup_memory(startup),
            'server_env': {k:v for k,v in env.items() if k not in VARIANT_ENV}, 'server_env_raw': env,
            'server_mounts': common_mounts, 'server_mounts_raw': mounts, 'source_manifest': sources,
            'gdn_compilation_config': compilation, 'gdn_readiness': readiness,
            'fi_control':fi_control, 'fi_evidence':fi_evidence,
            'source_validation':'missing_historical_snapshot' if legacy_sources else 'verified'}


def inductor_changes(base, candidate):
    if not base['fi_control'] or not candidate['fi_control']:
        return {'status':'not_captured_by_this_protocol', 'pinned':False}
    def group(run):
        result={}
        for entry in run['fi_evidence']['compiler_config_audit']['files']:
            key=entry.get('basename',Path(entry['path']).name)
            result.setdefault(key,[]).append(entry)
        return result
    a,b=group(base),group(candidate)
    changed={}
    for key in sorted(a.keys() & b.keys()):
        if sorted(x['selected_sha256'] for x in a[key]) != sorted(x['selected_sha256'] for x in b[key]):
            changed[key]={'baseline':a[key],'candidate':b[key]}
    return {'status':'observed_only', 'pinned':False,
            'baseline_files':sum(map(len,a.values())), 'candidate_files':sum(map(len,b.values())),
            'added':{k:b[k] for k in sorted(b.keys()-a.keys())},
            'removed':{k:a[k] for k in sorted(a.keys()-b.keys())},'changed':changed}


def compare(base, candidate, *, allow_legacy_sources=False):
    require(base['fi_control'] == candidate['fi_control'], 'controlled and unpinned FI runs are incompatible')
    if base['fi_control']:
        for key in ('import_mode','reference_sha256','reference_entries_sha256','runtime_source_sha256'):
            require(base['fi_evidence'][key] == candidate['fi_evidence'][key], 'different FI control '+key)
    for key in ('config', 'gpu_util'):
        require(base['config'][key] == candidate['config'][key], f'different {key}')
    for key in ('image', 'gpu', 'server_command', 'server_env', 'server_mounts', 'resolved_engine_config'):
        require(base[key] == candidate[key], f'incompatible {key}: baseline={base[key]!r}; candidate={candidate[key]!r}')
    legacy_sources = any(x['source_validation'] != 'verified' for x in (base,candidate))
    require(not legacy_sources or (allow_legacy_sources and not base['gdn_enabled']
            and not candidate['gdn_enabled'] and not base['fi_control']),
            'composed/controlled comparison requires both source manifests')
    a,b = base['source_manifest']['files'],candidate['source_manifest']['files']
    if not legacy_sources:
        changed = {p:{'baseline':a[p], 'candidate':b[p]} for p in a.keys() & b.keys() if a[p] != b[p]}
        require(not changed, 'different common runtime source hashes: ' + json.dumps(changed, sort_keys=True))
        original_gdn = 'build/vllm/' + GDN + GDN_FILES[0]
        allowed_a = source_extras(base['config']['variant']) | {original_gdn}
        allowed_b = source_extras(candidate['config']['variant']) | {original_gdn}
        require(a.keys()-b.keys() <= allowed_a and b.keys()-a.keys() <= allowed_b,
                'unrelated source-set changes: ' + str(sorted((a.keys()-b.keys())-allowed_a | (b.keys()-a.keys())-allowed_b)))
    return {'baseline_variant':base['config']['variant'], 'candidate_variant':candidate['config']['variant'],
            'baseline_moe_parent':base['variant_parent'], 'candidate_moe_parent':candidate['variant_parent'],
            'source_removed':{p:a[p] for p in sorted(a.keys()-b.keys())},
            'source_added':{p:b[p] for p in sorted(b.keys()-a.keys())},
            'intentional_environment_changes':{k:{'baseline':base['server_env_raw'].get(k), 'candidate':candidate['server_env_raw'].get(k)}
                for k in sorted(VARIANT_ENV) if base['server_env_raw'].get(k) != candidate['server_env_raw'].get(k)},
            'gdn_compilation':{'baseline':base['gdn_compilation_config'], 'candidate':candidate['gdn_compilation_config']},
            'fi_control':base['fi_control'],
            'fi_reference_sha256':base['fi_evidence']['reference_sha256'] if base['fi_control'] else None,
            'fi_import_mode':base['fi_evidence']['import_mode'] if base['fi_control'] else None,
            'inductor_pinned':False, 'inductor_config_changes':inductor_changes(base,candidate),
            'startup_memory_context':paired_startup_memory(base['startup_memory'],candidate['startup_memory']),
            'source_validation':'incomplete_historical_source_capture' if legacy_sources else 'verified'}
