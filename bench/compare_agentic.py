"""Strict CPU-only comparison of completed metric-version-2 agentic replay runs.

python3 -I bench/compare_agentic.py BASE CAND --output-prefix results/pair
python3 -I bench/compare_agentic.py --preflight

Requires captured production inputs and post-run agentic_input_verify.json.
Historical metric-version-1 or unhashed inputs are intentionally unsupported.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent

def module(name):
    spec=importlib.util.spec_from_file_location('_agentic_'+name,HERE/(name+'.py'))
    obj=importlib.util.module_from_spec(spec); spec.loader.exec_module(obj); return obj

IDENTITY=module('compare_runtime_identity')
INPUTS=module('agentic_input_manifest')
MTP=module('compare_greedy_decode')
require=IDENTITY.require
sha=IDENTITY.sha


def number(value, label, *, positive=False):
    require(type(value) in (int,float) and math.isfinite(value) and (value>0 if positive else value>=0), 'invalid '+label)
    return value


def integer(value, expected, label):
    require(type(value) is int and value==expected, 'incorrect '+label)


def close(actual, expected, label):
    if expected is None: require(actual is None, 'incorrect '+label)
    else:
        number(actual,label)
        require(math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-8), 'inconsistent '+label)


def validate_report(document, description):
    expected_concs=INPUTS.CONCURRENCIES[description['set']]
    require(isinstance(document,dict) and set(document)=={str(c) for c in expected_concs}, 'missing/extra replay concurrency')
    validated=[]
    for concurrency in expected_concs:
        result=document[str(concurrency)]; summary=result['summary']; rows=result['turns']
        integer(summary['metric_version'],2,'metric version')
        require(summary['valid'] is True and result['errors']==[], 'replay failed/incomplete')
        require(isinstance(rows,list), 'invalid turn collection')
        expected={}
        for copy_index in range((concurrency+7)//8):
            for trajectory in description['trajectories']:
                for turn in trajectory['turns']:
                    expected[(copy_index*8+trajectory['task'],turn['turn'])]=turn
        require(len(rows)==len(expected), 'missing/extra turn records')
        observed=set(); task_times={}; ttfts=[]; first=[]; later=[]; outputs=0; sources={}
        for row in rows:
            require(type(row['task']) is int and type(row['turn']) is int, 'invalid task/turn ID')
            key=(row['task'],row['turn'])
            require(key in expected and key not in observed, 'unexpected/duplicate task turn')
            observed.add(key); target=expected[key]
            for field,count in (('prompt',target['prompt_tokens']),('requested_completion',target['completion_tokens']),
                                ('completion',target['completion_tokens'])):
                integer(row[field],count,'turn '+field)
            require(row['success'] is True and row['stream_done'] is True and 'error' not in row, 'failed/truncated turn')
            require(row['finish_reason'] in ('length','stop') and row['completion_source'] in ('usage','length_finish'), 'unverified completion count')
            require(row['completion_source']!='length_finish' or row['finish_reason']=='length', 'length fallback without length finish')
            ttft=number(row['ttft'],'turn TTFT'); e2e=number(row['e2e'],'turn latency',positive=True)
            require(e2e>=ttft, 'TTFT exceeds full turn latency')
            ttfts.append(ttft); (first if row['turn']==0 else later).append(ttft)
            task_times[row['task']]=task_times.get(row['task'],0)+e2e
            outputs+=row['completion']; sources[row['completion_source']]=sources.get(row['completion_source'],0)+1
        wall=number(summary['wall_s'],'wall time',positive=True)
        require(wall+1e-6>=max(task_times.values()) and concurrency*wall+1e-6>=sum(task_times.values()), 'wall time cannot contain observed sequential turns')
        for field,expected_count in (('conc',concurrency),('agents',len(task_times)),('expected_agents',len(task_times)),
                                      ('turns',len(expected)),('expected_turns',len(expected)),('attempted_turns',len(expected)),
                                      ('failed_turns',0),('not_run_turns',0)):
            integer(summary[field],expected_count,'summary '+field)
        percentile=lambda values,p: 1000*sorted(values)[int(len(values)*p)] if values else None
        computed={'turns_per_s':len(expected)/wall,'out_tok_s':outputs/wall,
                  'ttft_first_turn_ms':1000*statistics.mean(first), 'ttft_p50_ms':percentile(ttfts,.5),
                  'ttft_later_p50_ms':percentile(later,.5),'ttft_later_p90_ms':percentile(later,.9),
                  'task_mean_s':statistics.mean(task_times.values())}
        for field,value in computed.items(): close(summary[field],value,'summary '+field)
        hit=number(summary['prefix_hit_rate'],'reported cache hit rate'); require(hit<=1,'reported cache hit rate exceeds one')
        validated.append({**summary,'set':description['set'],'output_tokens':outputs,
                          'ttft_p90_ms':percentile(ttfts,.9),'completion_sources':sources,
                          'cache_hit_validation':'reported rate only; raw per-concurrency counters absent'})
    return validated


def load(root, *, expected_input_sha256=INPUTS.PRODUCTION_SHA256):
    root=Path(root); hashes={}
    def data(relative):
        value=(root/relative).read_bytes(); hashes[relative]=sha(value); return value
    def text(relative):return data(relative).decode()
    def read(relative):return json.loads(text(relative))
    require(text('exit_code.txt').strip()=='0','failed/incomplete experiment')
    runtime=IDENTITY.load(root,text,read)
    require(runtime['config']['suite']=='agentic','not an agentic experiment')
    allowed_config={'tag','variant','suite','config','gpu_util','greedy_decode','nll_rounds','fi_control'}
    if runtime['fi_control']: allowed_config.add('fi_source_sha256')
    require(set(runtime['config'])==allowed_config,'unsupported/missing experiment config fields')
    require(runtime['config'].get('greedy_decode') in ('0','1') and runtime['config'].get('nll_rounds','').isdigit(), 'missing prior workload identity')
    manifest=read('agentic_input_manifest.json')
    require(manifest['sources']['bench/agentic_bench.py']==sha((HERE/'agentic_bench.py').read_bytes()),
            'unsupported agentic benchmark source version')
    verified=INPUTS.verify(root,read_bytes=data)
    recorded=read('agentic_input_verify.json')
    require(recorded=={**verified,'live_inputs_verified':True}, 'missing/failed post-run input/source verification')
    for relative,digest in manifest['sources'].items():
        require(runtime['source_manifest']['files'].get(relative)==digest, 'input helper/benchmark not captured in runtime source manifest: '+relative)
    rows=[]; mtp={}
    for kind in INPUTS.CONCURRENCIES:
        description=manifest['inputs'][kind]
        require(description['sha256']==expected_input_sha256[kind], 'not the fixed production trajectory bytes: '+kind)
        rows+=validate_report(read('agentic_'+kind+'.json'), {**description,'set':kind})
        mtp[kind]=MTP.mtp_delta(MTP.mtp_series(text(f'agentic_{kind}_metrics_before.txt')),
                                MTP.mtp_series(text(f'agentic_{kind}_metrics_after.txt')))
    return {**runtime,'input_manifest':manifest,'input_verification':recorded,'rows':rows,'mtp_by_set':mtp,'input_sha256':hashes}


def compare(base,candidate):
    runtime=IDENTITY.compare(base,candidate)
    for key in ('suite','greedy_decode','nll_rounds'):
        require(base['config'][key]==candidate['config'][key], 'different prior workload history: '+key)
    require(base['input_manifest']==candidate['input_manifest'], 'different agentic input identities/protocol/sources')
    rows=[]
    for a,b in zip(base['rows'],candidate['rows'],strict=True):
        require((a['set'],a['conc'],a['turns'],a['output_tokens'])==(b['set'],b['conc'],b['turns'],b['output_tokens']), 'different agentic replay scenarios')
        changes={key:100*(b[key]/a[key]-1) if a[key] else None for key in (
            'ttft_p50_ms','ttft_p90_ms','ttft_later_p50_ms','ttft_later_p90_ms','turns_per_s','out_tok_s','wall_s')}
        rows.append({'set':a['set'],'concurrency':a['conc'],'baseline':a,'candidate':b,'change_percent':changes,
                     'cache_hit_change_pp':100*(b['prefix_hit_rate']-a['prefix_hit_rate'])})
    analysis_files=('compare_agentic.py','agentic_input_manifest.py','compare_runtime_identity.py',
                    'compare_greedy_decode.py','serving_matrix.py')
    return {'status':'complete_compatible_pair','metric_version':2,'baseline':base,'candidate':candidate,
            'rows':rows,'runtime_changes':runtime,
            'mtp_by_set':{kind:{'baseline':base['mtp_by_set'][kind],'candidate':candidate['mtp_by_set'][kind],
                'scope':'entire replay set, all listed concurrencies; not a per-concurrency metric'} for kind in INPUTS.CONCURRENCIES},
            'analysis_source_sha256':{name:sha((HERE/name).read_bytes()) for name in analysis_files},
            'limitations':[
                'Replay fixes prerecorded prompt tokens and completion lengths; it does not measure live agent quality or generated-text equality.',
                'TTFT p90 over all turns is derived from detailed records; percentile uses sorted_values[int(n*p)], matching existing p50/later-p90 semantics.',
                'Cache-hit rates are validated as finite reported fractions; raw counters per concurrency are absent, so those ratios cannot be independently recomputed.',
                'MTP acceptance uses runner before/after counter deltas for each complete short/long set; no per-concurrency attribution.',
                'Input identity is established by captured bytes and post-run source/input verification; detailed reports contain lengths, not echoed prompt IDs or cache salts.',
                'Cache salts use process IDs. Distinct fresh servers isolate paired runs; within-run PID namespaces and executed replay argv are not separately recorded.',
                'Fixed FI replay checks saved reference tactics through shutdown when enabled; Inductor choices are observed, not pinned. A single sequential pair is not a confidence interval.']}


def markdown(report):
    lines=[f"Baseline `{report['baseline']['name']}`; candidate `{report['candidate']['name']}`. Completed metric-version-2 replay; fixed production input hashes and runtime identity verified.", '',
           '| set / concurrency | turns / output tokens | TTFT p50 ms base / cand | TTFT p90 ms base / cand | turns/s base / cand | output tok/s base / cand | output change | cache hit % base / cand | wall s base / cand |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in report['rows']:
        a,b=row['baseline'],row['candidate']
        pair=lambda key: f'{a[key]:.2f} / {b[key]:.2f}'
        lines.append(f"| {row['set']} c{row['concurrency']} | {a['turns']} / {a['output_tokens']} | {pair('ttft_p50_ms')} | {pair('ttft_p90_ms')} | {pair('turns_per_s')} | {pair('out_tok_s')} | {row['change_percent']['out_tok_s']:+.2f}% | {100*a['prefix_hit_rate']:.2f} / {100*b['prefix_hit_rate']:.2f} | {pair('wall_s')} |")
    lines += ['', '| MTP set aggregate | draft steps base / cand | drafted tokens base / cand | accepted tokens base / cand | acceptance % base / cand | acceptance length base / cand |', '|---|---:|---:|---:|---:|---:|']
    for kind,entry in report['mtp_by_set'].items():
        a,b=entry['baseline'],entry['candidate']
        lines.append(f"| {kind} | {a['num_drafts']} / {b['num_drafts']} | {a['num_draft_tokens']} / {b['num_draft_tokens']} | {a['num_accepted_tokens']} / {b['num_accepted_tokens']} | {100*a['acceptance_rate']:.2f} / {100*b['acceptance_rate']:.2f} | {a['acceptance_length']:.3f} / {b['acceptance_length']:.3f} |")
    lines+=['','Acceptance = accepted/drafted tokens; acceptance length = 1 + accepted tokens/draft steps. These aggregate all concurrencies within a set.','']
    lines+=['- '+item for item in report['limitations']]
    lines+=['',IDENTITY.format_startup_memory(report['runtime_changes']['startup_memory_context']),
            '','Validated runtime changes: '+json.dumps(report['runtime_changes'],sort_keys=True)]
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('baseline',nargs='?',type=Path)
    parser.add_argument('candidate',nargs='?',type=Path); parser.add_argument('--output-prefix',type=Path)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--production-inputs',action='store_true',help='also hash and validate the actual production inputs during CPU preflight')
    args=parser.parse_args()
    if args.preflight:
        tests=module('test_compare_agentic'); report=tests.preflight()
        if args.production_inputs: report['production_input_cpu_validation']=tests.production_preflight(HERE.parent)
        paths=('compare_agentic.py','agentic_input_manifest.py','test_compare_agentic.py','AGENTIC_COMPARISON.md',
               'compare_runtime_identity.py','compare_greedy_decode.py','serving_matrix.py','agentic_bench.py')
        report['source_sha256']={'bench/'+name:sha((HERE/name).read_bytes()) for name in paths}
        print(json.dumps(report,indent=2,allow_nan=False)); return
    if args.production_inputs: parser.error('--production-inputs requires --preflight')
    if args.baseline is None or args.candidate is None or args.output_prefix is None:
        parser.error('provide BASE CAND --output-prefix FILE, or --preflight')
    report=compare(load(args.baseline),load(args.candidate)); prefix=args.output_prefix
    prefix.parent.mkdir(parents=True,exist_ok=True)
    for suffix,value in (('.json',json.dumps(report,indent=2,allow_nan=False)+'\n'),('.md',markdown(report))):
        target=Path(str(prefix)+suffix)
        with target.open('x') as handle:handle.write(value)
    print(json.dumps({'status':report['status'],'rows':len(report['rows']),'output_prefix':str(prefix)}))

if __name__=='__main__':main()
