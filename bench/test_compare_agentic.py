"""CPU-only artifact fixtures for the final agentic comparator; no HTTP or GPU."""
import copy
import importlib.util
import json
import lzma
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent

def module(name):
    spec = importlib.util.spec_from_file_location('_agentic_test_' + name, HERE / (name + '.py'))
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj)
    return obj


def preflight():
    compare, inputs, benchmark = map(module, ('compare_agentic', 'agentic_input_manifest', 'agentic_bench'))
    identity = compare.IDENTITY
    rejected = []
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary); repo = folder / 'repo'; repo.mkdir()
        for name in ('agentic_bench.py', 'agentic_input_manifest.py'):
            p = repo / 'bench' / name; p.parent.mkdir(exist_ok=True); p.write_bytes((HERE/name).read_bytes())
        documents = {}
        for kind in inputs.CONCURRENCIES:
            documents[kind] = {'trajectories': [{'task': task, 'turns': [
                {'prompt_ids': [1, task, turn, 12], 'completion_tokens': 3 + turn,
                 'live_s': 100.0} for turn in range(2)]} for task in range(8)]}
            p = repo / inputs.source_path(kind); p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(lzma.compress(json.dumps(documents[kind]).encode()))
        roots = [folder / 'base', folder / 'candidate']
        def save(root, relative, value):
            p = root / relative; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(value if isinstance(value, str) else json.dumps(value))
        for root in roots:
            root.mkdir(); inputs.snapshot(repo, root)
            save(root, 'agentic_input_verify.json', inputs.verify(root, repo))
            variant = 'next' if root.name == 'base' else 'shared_exact'
            save(root, 'exit_code.txt', '0\n')
            save(root, 'config.txt', f'tag={root.name}\nvariant={variant}\nsuite=agentic\nconfig=next\ngpu_util=0.78\ngreedy_decode=0\nnll_rounds=0\nfi_control=0\n')
            save(root, 'image.txt', 'fixture-image'); save(root, 'commit.txt', 'fixture')
            save(root, 'uncommitted.diff', '')
            save(root, 'gpu_before.csv', 'name,uuid,driver_version,memory.total [MiB],power.limit [W]\nSM120,GPU-fixture,1,97887 MiB,600 W\n')
            command = ['serve', 'fixture', '--speculative-config', '{"method":"mtp","num_speculative_tokens":2}']
            env = ['COMMON=1', 'VLLM_CACHE_ROOT=/root/.cache/vllm/qwen-sequential-' + variant]
            if variant != 'next':
                command += ['--moe-backend', 'flashinfer_b12x']; env += ['VLLM_B12X_FIX_SCALES=1', 'VLLM_B12X_SHARED_WORKSPACE=1']
            save(root, 'startup.log', 'Initializing a V1 LLM engine (v0.32.0) with config: ' +
                 "compilation_config={'splitting_ops': " + repr(identity.STANDARD_SPLITS+identity.AUTO_SPLITS) +
                 ", 'pass_config': {'fuse_rope_kvcache_cat_mla': False}}, kernel_config=KernelConfig(moe_backend='" +
                 ('auto' if variant == 'next' else 'flashinfer_b12x') + "')")
            files = {name: 'fixture\n' for name in identity.source_extras(variant) | set(identity.expected_mounts(variant).values()) | {
                'scripts/serve.sh', 'scripts/cudagraph_sizes.txt', 'bench/qwen_experiment.sh',
                'bench/serving_matrix.py', 'bench/source_manifest.py'}}
            files.update({'bench/' + name: (HERE/name).read_text() for name in ('agentic_bench.py', 'agentic_input_manifest.py')})
            for name, value in files.items(): save(root, 'source_files/' + name, value)
            save(root, 'source_manifest.json', {'schema_version':1, 'variant':variant, 'fi_control':False,
                'files':{name:compare.sha(value.encode()) for name,value in files.items()}})
            save(root, 'source_verify.log', {'status':'passed', 'action':'verify', 'variant':variant, 'files':len(files)})
            mounts = [{'Type':'bind','Source':'/fixture/' + src,'Destination':dst,'RW':False,'Mode':'ro'}
                      for dst,src in identity.expected_mounts(variant).items()]
            mounts += [{'Type':'bind','Source':'/fixture/results/qwen_sequential/' + root.name + '/serving' if dst == '/bench-results' else '/fixture' + dst,
                        'Destination':dst,'RW':True,'Mode':''} for dst in ('/bench-results','/root/.cache/vllm','/root/.cache/flashinfer','/root/.cache/huggingface')]
            save(root, 'container.json', [{'Name':'/qwen-exp-' + root.name,'Image':'fixture-image','Config':{'Cmd':command,'Env':env},'Mounts':mounts}])
            for kind, concs in inputs.CONCURRENCIES.items():
                report = {}
                for concurrency in concs:
                    rows, expected = [], {}
                    for n in range((concurrency + 7)//8):
                        for task in documents[kind]['trajectories']:
                            task_id = n*8 + task['task']; expected[task_id] = len(task['turns'])
                            for turn, request in enumerate(task['turns']):
                                rows.append({'task':task_id,'turn':turn,'prompt':len(request['prompt_ids']),
                                    'requested_completion':request['completion_tokens'],'completion':request['completion_tokens'],
                                    'completion_source':'usage','ttft':0.01 + .001*turn,'finish_reason':'length',
                                    'stream_done':True,'success':True,'e2e':0.02})
                    report[str(concurrency)] = {'summary':benchmark._replay_summary(concurrency,expected,rows,10.0,.5,True), 'turns':rows,'errors':[]}
                save(root, f'agentic_{kind}.json', report)
                for phase, numbers in (('before',(100,200,100)),('after',(110,220,112))):
                    save(root, f'agentic_{kind}_metrics_{phase}.txt', '\n'.join(
                        f'vllm:spec_decode_{name}_total{{model_name="bench"}} {value}'
                        for name,value in zip(compare.MTP.COUNTERS,numbers)))
        fixture_hashes={kind:compare.sha((repo/inputs.source_path(kind)).read_bytes()) for kind in inputs.CONCURRENCIES}
        load=lambda root:compare.load(root,expected_input_sha256=fixture_hashes)
        baseline, candidate = map(load, roots)
        try: compare.load(roots[0])
        except ValueError: rejected.append('nonproduction_inputs')
        else: raise AssertionError('accepted synthetic inputs as production')
        result = compare.compare(baseline, candidate)
        assert len(result['rows']) == 8 and all(row['change_percent']['out_tok_s'] == 0 for row in result['rows'])
        assert result['mtp_by_set']['short']['baseline']['acceptance_rate'] == .6
        assert result['rows'][0]['baseline']['ttft_p90_ms'] == 11.0
        assert '0.00%' in compare.markdown(result)
        def altered(label, relative, action):
            p=roots[1]/relative; original=p.read_bytes()
            value=json.loads(original) if p.suffix == '.json' else original.decode()
            changed=action(copy.deepcopy(value)); save(roots[1],relative,changed)
            try:
                try: compare.compare(load(roots[0]), load(roots[1]))
                except (ValueError, KeyError, TypeError): rejected.append(label)
                else: raise AssertionError('accepted invalid fixture: ' + label)
            finally: p.write_bytes(original)
        def mutate_row(field, value):
            def mutate(document): document['1']['turns'][0][field]=value; return document
            return mutate
        def mutate_summary(field, value):
            def mutate(document): document['1']['summary'][field]=value; return document
            return mutate
        for field,value in [('metric_version',1),('valid',False),('turns',15),('failed_turns',1),
                            ('out_tok_s',99),('ttft_later_p90_ms',99),('prefix_hit_rate',1.1),('wall_s',float('inf'))]:
            altered('summary_' + field,'agentic_short.json',mutate_summary(field,value))
        for field,value in [('completion',1),('requested_completion',1),('prompt',1),('turn',99),
                            ('task',99),('success',False),('stream_done',False),('completion_source','unknown'),
                            ('ttft',float('nan')),('e2e',.001),('error','failure')]:
            altered('turn_' + field,'agentic_short.json',mutate_row(field,value))
        altered('missing_concurrency','agentic_long.json',lambda x:{k:v for k,v in x.items() if k != '32'})
        altered('extra_concurrency','agentic_short.json',lambda x:{**x,'2':x['1']})
        def duplicate(x): x['1']['turns'][1]=x['1']['turns'][0]; return x
        altered('duplicate_turn','agentic_short.json',duplicate)
        def failed(x): x['1']['errors']=['failure']; return x
        altered('error_list','agentic_short.json',failed)
        altered('failed_experiment','exit_code.txt',lambda x:'1')
        altered('source_changed','source_files/bench/agentic_bench.py',lambda x:x+'# changed\n')
        altered('different_history','config.txt',lambda x:x.replace('nll_rounds=0','nll_rounds=5'))
        altered('unknown_config','config.txt',lambda x:x+'new_runtime_control=1\n')
        altered('missing_input_verification','agentic_input_verify.json',lambda x:{**x,'live_inputs_verified':False})
        altered('changed_input_metadata','agentic_input_manifest.json',lambda x:{**x,'schema_version':9})
        altered('changed_input_source','agentic_inputs/source_files/bench/agentic_bench.py',lambda x:x+'# changed\n')
        altered('mtp_reset','agentic_short_metrics_after.txt',lambda x:x.replace(' 110',' 90'))
        altered('mtp_missing','agentic_short_metrics_after.txt',lambda x:'')
        altered('mtp_changed_labels','agentic_short_metrics_after.txt',lambda x:x.replace('bench','other'))
        altered('unexpected_mtp_acceptance','agentic_short_metrics_after.txt',lambda x:x.replace(' 112',' 999'))
        p=roots[1]/'agentic_inputs/trajectories_short_production.json.xz'; raw=p.read_bytes()
        try:
            p.write_bytes(raw+b'tamper')
            try: load(roots[1])
            except (ValueError,lzma.LZMAError): rejected.append('changed_input_bytes')
            else: raise AssertionError('accepted modified input bytes')
        finally:p.write_bytes(raw)
        # Matching lengths alone must not make differing prompt IDs compatible.
        changed=copy.deepcopy(candidate)
        changed['input_manifest']['inputs']['short']['replay_sha256']='0'*64
        try:compare.compare(baseline,changed)
        except ValueError:rejected.append('different_prompt_token_identity')
        else:raise AssertionError('accepted different prompt identities')
    return {'status':'passed','cpu_only':True,'synthetic_schema':'actual agentic_bench._replay_summary metric_version=2',
            'valid_pair_rows':8,'rejected_fixtures':rejected,'rejected_count':len(rejected),
            'runtime_identity_gates':'reused compare_runtime_identity; composed/FI gates have their existing CPU preflight',
            'gpu_or_http_used':False}

def production_preflight(root):
    """Validate production bytes and all eight replay shapes without serving them."""
    compare, helper, benchmark = map(module, ('compare_agentic', 'agentic_input_manifest', 'agentic_bench'))
    root=Path(root)
    with tempfile.TemporaryDirectory(prefix='agentic-input-cpu-') as folder:
        output=Path(folder)
        manifest=helper.snapshot(root,output)
        verification=helper.verify(output,root)
        actual={}
        for kind,entry in manifest['inputs'].items():
            assert entry['sha256']==helper.PRODUCTION_SHA256[kind]
            document={}
            for concurrency in helper.CONCURRENCIES[kind]:
                rows=[]; expected={}
                for copy_index in range((concurrency+7)//8):
                    for task in entry['trajectories']:
                        task_id=copy_index*8+task['task']; expected[task_id]=task['turn_count']
                        for turn in task['turns']:
                            rows.append({'task':task_id,'turn':turn['turn'],'prompt':turn['prompt_tokens'],
                                'requested_completion':turn['completion_tokens'],'completion':turn['completion_tokens'],
                                'completion_source':'usage','ttft':.01,'e2e':.02,'finish_reason':'length',
                                'stream_done':True,'success':True})
                document[str(concurrency)]={'summary':benchmark._replay_summary(concurrency,expected,rows,100.0,.5,True),
                                            'turns':rows,'errors':[]}
            validated=compare.validate_report(document,{**entry,'set':kind})
            actual[kind]={key:entry[key] for key in ('sha256','size_bytes','replay_sha256','base_turns','base_output_tokens','cases')}
            actual[kind]['per_trajectory']=[{'task':x['task'],'turns':x['turn_count'],'output_tokens':x['output_tokens']} for x in entry['trajectories']]
            actual[kind]['validated_rows']=len(validated)
        return {'status':'passed','input_verification':verification,'inputs':actual,
                'timings':'synthetic CPU fixtures only, generated with actual production input lengths; no measured performance'}


if __name__ == '__main__':
    print(json.dumps(preflight(),indent=2))
