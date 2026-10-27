"""CPU-only immutable-input snapshot for final agentic replay comparisons.

snapshot --root REPO --output RUN copies both production .xz files before replay.
verify --root REPO --output RUN verifies live inputs, snapshots, and source bytes.
Replay RUN/agentic_inputs/trajectories_{short,long}_production.json.xz.
"""
import argparse
import hashlib
import json
import lzma
from pathlib import Path

CONCURRENCIES = {'short':[1,4,8,16,32], 'long':[1,8,32]}
PRODUCTION_SHA256 = {
    'short':'e24df8aa498518add17b8188ef834ea4272ca78991513f6bbe0118f38021c680',
    'long':'3852ffe3133be6ad2c93442682906f45990944f86e4126ff10f9c7cbf83580d8'}
SOURCES = ('bench/agentic_bench.py', 'bench/agentic_input_manifest.py')
PROTOCOL = {'metric_version':2, 'model':'bench', 'temperature':0, 'min_tokens_equals_max_tokens':True,
            'ignore_eos':True, 'stream':True, 'include_usage':True,
            'copies':'ceil(concurrency/8)', 'task_id':'copy_index*8+recorded_task_id',
            'cache_salt':'agentic-c{concurrency}-{process_id}-copy{copy_index}',
            'concurrencies':CONCURRENCIES, 'replay_inputs':'captured_xz_copies'}


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(data): return hashlib.sha256(data).hexdigest()


def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def source_path(kind): return f'results/agentic/trajectories_{kind}_production.json.xz'


def snapshot_path(kind): return f'agentic_inputs/trajectories_{kind}_production.json.xz'


def describe(kind, data):
    document = json.loads(lzma.decompress(data))
    trajectories = document['trajectories']
    require(isinstance(trajectories,list) and [t['task'] for t in trajectories] == list(range(8))
            and all(type(t['task']) is int for t in trajectories), 'expected eight ordered production task IDs')
    described=[]
    for task in trajectories:
        require(isinstance(task['turns'],list) and task['turns'], 'empty trajectory')
        turns=[]
        for index, turn in enumerate(task['turns']):
            tokens, completion = turn['prompt_ids'], turn['completion_tokens']
            require(isinstance(tokens,list) and tokens and all(type(x) is int and x >= 0 for x in tokens), 'invalid prompt token IDs')
            require(type(completion) is int and completion > 0, 'invalid requested completion length')
            turns.append({'turn':index, 'prompt_tokens':len(tokens), 'prompt_ids_sha256':sha(canonical(tokens)),
                          'completion_tokens':completion})
        described.append({'task':task['task'], 'turns':turns, 'turn_count':len(turns),
                          'output_tokens':sum(t['completion_tokens'] for t in turns)})
    total_turns=sum(t['turn_count'] for t in described); total_outputs=sum(t['output_tokens'] for t in described)
    cases=[{'concurrency':c, 'copies':(c+7)//8, 'agents':8*((c+7)//8),
            'turns':total_turns*((c+7)//8), 'output_tokens':total_outputs*((c+7)//8)} for c in CONCURRENCIES[kind]]
    return {'source_path':source_path(kind), 'snapshot_path':snapshot_path(kind), 'sha256':sha(data),
            'size_bytes':len(data), 'replay_sha256':sha(canonical(described)), 'trajectories':described,
            'base_turns':total_turns, 'base_output_tokens':total_outputs, 'cases':cases}


def snapshot(root, output):
    root, output=Path(root),Path(output)
    require(not (output/'agentic_input_manifest.json').exists() and not (output/'agentic_inputs').exists(),
            'agentic input snapshot already exists; do not overwrite an experiment')
    payloads={snapshot_path(kind):(root/source_path(kind)).read_bytes() for kind in CONCURRENCIES}
    manifest={'schema_version':1, 'protocol':PROTOCOL,
              'inputs':{kind:describe(kind,payloads[snapshot_path(kind)]) for kind in CONCURRENCIES}, 'sources':{}}
    for relative in SOURCES:
        data=(root/relative).read_bytes(); manifest['sources'][relative]=sha(data)
        payloads['agentic_inputs/source_files/'+relative]=data
    for relative,data in payloads.items():
        path=output/relative; path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as handle: handle.write(data)
    temporary=output/'agentic_input_manifest.json.tmp'
    with temporary.open('x') as handle: handle.write(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    temporary.replace(output/'agentic_input_manifest.json')
    return manifest


def verify(output, root=None, *, read_bytes=None):
    output=Path(output); root=Path(root) if root is not None else None
    read_bytes=read_bytes or (lambda relative:(output/relative).read_bytes())
    raw=read_bytes('agentic_input_manifest.json'); manifest=json.loads(raw)
    require(manifest['schema_version'] == 1 and manifest['protocol'] == PROTOCOL, 'unsupported input manifest protocol')
    require(set(manifest['inputs']) == set(CONCURRENCIES) and set(manifest['sources']) == set(SOURCES), 'input manifest set mismatch')
    for kind in CONCURRENCIES:
        entry=manifest['inputs'][kind]; data=read_bytes(snapshot_path(kind))
        require(sha(data)==entry['sha256'], 'captured input byte hash mismatch: '+kind)
        require(describe(kind,data)==entry, 'captured input metadata differs: '+kind)
        if root is not None:
            require(sha((root/source_path(kind)).read_bytes())==entry['sha256'], 'live input changed: '+kind)
    for relative,digest in manifest['sources'].items():
        require(sha(read_bytes('agentic_inputs/source_files/'+relative))==digest, 'input helper/benchmark snapshot changed')
        if root is not None: require(sha((root/relative).read_bytes())==digest, 'live input helper/benchmark source changed')
    require(manifest['sources']['bench/agentic_input_manifest.py']==sha(Path(__file__).read_bytes()), 'unsupported input helper source version')
    return {'status':'passed', 'action':'verify', 'schema_version':1, 'manifest_sha256':sha(raw),
            'live_inputs_verified':root is not None, 'inputs':{k:v['sha256'] for k,v in manifest['inputs'].items()},
            'replay_sha256':{k:v['replay_sha256'] for k,v in manifest['inputs'].items()}, 'sources':manifest['sources']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('snapshot','verify')); parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path); args=parser.parse_args()
    if args.action=='snapshot':
        result=snapshot(args.root,args.output)
        print(json.dumps({'status':'passed','action':'snapshot','inputs':{k:v['sha256'] for k,v in result['inputs'].items()}}))
    else: print(json.dumps(verify(args.output,args.root)))

if __name__=='__main__': main()
