"""CPU-only startup-memory parsing regressions; fixtures match saved vLLM 0.32 logs."""
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
FIXTURE='''(EngineCore pid=400) INFO [model_runner.py:1127] Graph capturing finished in 19 secs, took 4.42 GiB
(EngineCore pid=400) INFO [gpu_worker.py:759] Available KV cache memory: 45.5 GiB
(EngineCore pid=400) INFO [kv_cache_utils.py:2327] GPU KV cache size: 3,437,190 tokens, Maximum concurrency for 131,072 tokens per request: 26.22x
(EngineCore pid=400) INFO [model_runner.py:1127] Graph capturing finished in 17 secs, took 3.45 GiB
(EngineCore pid=400) INFO [gpu_worker.py:971] CUDA graph pool memory: 3.45 GiB (actual), 4.46 GiB (estimated), difference: 1.01 GiB (29.3%).
'''


def preflight():
    spec=importlib.util.spec_from_file_location('memory_identity',HERE/'compare_runtime_identity.py')
    identity=importlib.util.module_from_spec(spec);spec.loader.exec_module(identity)
    actual=identity.parse_startup_memory(FIXTURE)
    assert actual['status']=='complete'
    assert actual['kv_available_gib']==45.5 and actual['kv_capacity_tokens']==3437190
    assert actual['concurrency_request_tokens']==131072 and actual['maximum_concurrency']==26.22
    assert actual['cuda_graph_actual_gib']==3.45 and actual['cuda_graph_estimated_gib']==4.46
    assert actual['graph_capture_events_gib']==[4.42,3.45]
    missing=identity.parse_startup_memory('fixture engine without memory logs')
    assert missing['status']=='missing' and len(missing['missing_fields'])==6
    partial=identity.parse_startup_memory('Available KV cache memory: 45.5 GiB\n')
    assert partial['status']=='partial' and len(partial['missing_fields'])==5
    other=identity.parse_startup_memory(FIXTURE.replace('45.5 GiB','45.64 GiB').replace('3,437,190','3,447,538').replace('26.22x','26.30x').replace('3.45 GiB','3.30 GiB').replace('4.46 GiB','4.33 GiB'))
    pair=identity.paired_startup_memory(actual,other)
    assert pair['delta_candidate_minus_baseline']['kv_capacity_tokens']==10348
    assert abs(pair['delta_candidate_minus_baseline']['cuda_graph_actual_gib']+.15)<1e-9
    assert '45.50 / 45.64' in identity.format_startup_memory(pair)
    assert 'not recorded' in identity.format_startup_memory(identity.paired_startup_memory(missing,partial))
    zeros=identity.parse_startup_memory('CUDA graph pool memory: 0 GiB (actual), 0 GiB (estimated)\n')
    assert zeros['cuda_graph_actual_gib']==0
    ansi=identity.parse_startup_memory('\x1b[32m'+FIXTURE+'\x1b[0m')
    assert ansi['kv_capacity_tokens']==actual['kv_capacity_tokens']
    rejected=[]
    mutations={
        'duplicate_available':FIXTURE+'Available KV cache memory: 45.5 GiB\n',
        'duplicate_capacity':FIXTURE+'GPU KV cache size: 3,437,190 tokens\n',
        'duplicate_concurrency':FIXTURE+'Maximum concurrency for 131,072 tokens per request: 26.22x\n',
        'duplicate_final_graph':FIXTURE+'CUDA graph pool memory: 3.45 GiB (actual), 4.46 GiB (estimated)\n',
        'same_line_duplicate_graph':FIXTURE.replace('CUDA graph pool memory:', 'CUDA graph pool memory: 3.45 GiB (actual), 4.46 GiB (estimated); CUDA graph pool memory:'),
        'nan_available':FIXTURE.replace('45.5 GiB','nan GiB'),
        'infinite_graph':FIXTURE.replace('4.46 GiB','inf GiB'),
        'negative_graph':FIXTURE.replace('4.46 GiB','-1 GiB'),
        'negative_available':FIXTURE.replace('45.5 GiB','-1 GiB'),
        'zero_available':FIXTURE.replace('45.5 GiB','0 GiB'),
        'float_capacity':FIXTURE.replace('3,437,190','3437190.5'),
        'bad_comma_capacity':FIXTURE.replace('3,437,190','34,371,90'),
        'zero_capacity':FIXTURE.replace('3,437,190','0'),
        'zero_request_tokens':FIXTURE.replace('131,072','0'),
        'negative_concurrency':FIXTURE.replace('26.22x','-26.22x'),
        'inconsistent_concurrency':FIXTURE.replace('26.22x','35.22x'),
        'inconsistent_final_graph':FIXTURE.replace('17 secs, took 3.45','17 secs, took 3.60'),
        'malformed_known_record':FIXTURE.replace('Available KV cache memory: 45.5 GiB','Available KV cache memory: unavailable'),
        'missing_estimated_number':FIXTURE.replace('4.46 GiB (estimated)','GiB (estimated)'),
    }
    for name,log in mutations.items():
        try:identity.parse_startup_memory(log)
        except ValueError:rejected.append(name)
        else:raise AssertionError('invalid memory fixture accepted: '+name)
    return {'status':'passed','cpu_only':True,'rejected_fixtures':rejected,'rejected_count':len(rejected),
            'checks':['final actual separate from profiling capture','explicit partial/missing fields',
                      'capacity differences reported without equality gate','zero graph pool valid','ANSI logs'],
            'fixture_memory':actual,'example_memory':other}

if __name__=='__main__':print(json.dumps(preflight(),indent=2))
