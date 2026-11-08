#!/usr/bin/env python3
"""Strict controlled-PDL smoke comparison; not a throughput/quality verdict."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import tempfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
P=module('pin_provenance',HERE/'pdl_pin_provenance.py')
I=module('runtime_identity',ROOT/'bench/compare_runtime_identity.py')
E=module('pin_events',HERE/'verify_pin_events.py')
N=module('nll_compare',ROOT/'eval/repeated_nll.py')
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(data):return hashlib.sha256(data).hexdigest()
def pin_singleton_compiler_evidence(root,verification):
 require(not verification.get('compiler_config_audit_error'),'FI compiler audit failed')
 require(verification.get('selected_inductor_config_count')==0,'empty audit with nonzero count')
 events=E.verify(root)
 require(events['mode']=='pin','empty best_config audit without config pin')
 aot=json.loads((root/'pin_aot_verify.json').read_text())
 require(aot['status']=='passed_model_bound_autotuners' and aot['mode']=='pin','missing pinned AOT binding')
 require(aot['event_report_sha256']==sha((root/'pin_events_verify.json').read_bytes()),'AOT event binding differs')
 reference=json.loads((root/'serving/pin_reference.json').read_text())['reference']
 require(aot['records'] and aot['model_identities'],'empty AOT config coverage')
 for row in aot['records']:
  key=row['identity'];require(key in reference and len(reference[key])==1,'unknown AOT identity')
  expected=reference[key][0]['config']
  require(row['compiled_configs']==[expected],'AOT singleton config differs')
  require(events['launched_configs'].get(key) and all(v==expected for v in events['launched_configs'][key]),'actual singleton launch differs')

def install_pin_fi_contract(identity_module):
 # Replace exactly the nonempty .best_config assumption; all FI checks remain.
 old="    require(compiler['files'] and not verification.get('compiler_config_audit_error'), 'FI compiler config audit missing/failed')"
 source=inspect.getsource(identity_module.verify_fi)
 require(source.count(old)==1,'shared FI compiler audit contract changed')
 new="    if not compiler['files']:\n        pin_singleton_compiler_evidence(root,verification)\n    else:\n"+old.replace('    require','        require',1)
 scope=dict(identity_module.__dict__,pin_singleton_compiler_evidence=pin_singleton_compiler_evidence)
 exec(compile(source.replace(old,new),'<local_pinned_fi_contract>','exec'),scope)
 identity_module.verify_fi=scope['verify_fi']

install_pin_fi_contract(I)

def normalize(root,target,mode):
 require((root/'exit_code.txt').read_text().strip()=='0','incomplete/failed pin run')
 snapshot=json.loads((root/'pdl_source_manifest.json').read_text())
 if 'run_vllm_pdl_pin_focused.sh' in snapshot['files']:
  module('focused_pin_provenance',HERE/'pdl_pin_focused_provenance.py').run('verify',root,mode)
  require((root/'cleanup_status.txt').read_text().strip()=='0','focused cleanup failed')
  require(not (root/'container_after.txt').read_text().strip() and not (root/'gpu_after.txt').read_text().strip(),'focused owned-container/GPU vacancy missing')
  review=module('focused_review_guard',HERE/'pdl_focused_freeze.py').verify(root/'focused_review.json','pin',root/'serving/pin_reference.json')
  require(json.loads((root/'focused_review_verify.json').read_text())==review,'focused reviewed-source verification differs')
 else:P.run('verify',root,mode)
 events=E.verify(root)
 require(events['mode']=='pin','not a pin run')
 aot=json.loads((root/'pin_aot_verify.json').read_text())
 require(aot['status']=='passed_model_bound_autotuners' and aot['mode']=='pin','missing pinned AOT gate')
 require(aot['analysis_sources']=={name:sha((HERE/name).read_bytes()) for name in ('verify_pin_aot.py','pin_bundle_binding.py','symbolic_pickle.py','pin_inductor_core.py')},'stale AOT analysis sources')
 require(aot['event_report_sha256']==sha((root/'pin_events_verify.json').read_bytes()),'stale AOT event binding')
 require(json.loads((root/'pin_events_verify.json').read_text())==events,'recorded event verification differs')
 shutil.copytree(root,target,ignore=shutil.ignore_patterns('pdl_cache'))
 data=json.loads((target/'container.json').read_text());c,=data
 require(c['Name']=='/qwen-pdl-'+root.name,'container name mismatch');c['Name']='/qwen-exp-'+root.name
 env=dict(x.split('=',1) for x in c['Config']['Env']);require(len(env)==len(c['Config']['Env']),'duplicate env key')
 expected={'TORCHINDUCTOR_ENABLE_PDL':str(mode),'TORCHINDUCTOR_CACHE_DIR':'/bench-results/pdl_cache/inductor',
  'TRITON_CACHE_DIR':'/bench-results/pdl_cache/triton','QWEN_INDUCTOR_PIN_MODE':'pin',
  'QWEN_INDUCTOR_PIN_OUT':'/bench-results/pin_control','QWEN_INDUCTOR_PIN_REFERENCE':'/bench-results/pin_reference.json',
  'QWEN_INDUCTOR_PIN_REFERENCE_SHA256':sha((root/'serving/pin_reference.json').read_bytes())}
 for k,v in expected.items():require(env.pop(k,None)==v,'unexpected env: '+k)
 require(env['PYTHONPATH']=='/opt/pdl-pin/pin_bootstrap:/opt/qwen-fi-control','bootstrap path mismatch')
 env['PYTHONPATH']='/opt/qwen-fi-control'
 require(env['VLLM_CACHE_ROOT']=='/bench-results/pdl_cache/vllm','cache root mismatch')
 env['VLLM_CACHE_ROOT']='/root/.cache/vllm/qwen-sequential-scalar';c['Config']['Env']=[k+'='+v for k,v in env.items()]
 source_mount,=[m for m in c['Mounts'] if m['Destination']=='/opt/pdl-pin']
 require(Path(source_mount['Source']).resolve()==HERE and source_mount['RW'] is False and source_mount['Mode']=='ro','pin source mount mismatch')
 c['Mounts'].remove(source_mount)
 result_mount,=[m for m in c['Mounts'] if m['Destination']=='/bench-results']
 require(Path(result_mount['Source']).resolve()==root/'serving','result mount mismatch')
 result_mount['Source']=str(ROOT/'results/qwen_sequential'/root.name/'serving')
 (target/'container.json').write_text(json.dumps(data))
 for path in (target/'serving').rglob('*'):
  if path.is_file() and path.suffix in ('.json','.jsonl','.log','.txt') and 'serving_matrix_' in str(path):
   text=path.read_text();old='qwen-pdl-'+root.name;new='qwen-exp-'+root.name
   if old in text:path.write_text(text.replace(old,new))
 hashes={}
 def text(name):
  data=(target/name).read_bytes();hashes[name]=sha(data);return data.decode()
 def read(name):return json.loads(text(name))
 runtime=I.load(target,text,read)
 return runtime,events,aot,{'container_original_sha256':sha((root/'container.json').read_bytes()),
  'allowed_normalization':'Validated PDL/cache/pin-hook env, source mount, result path and container prefix adapted to common FI identity contract. All original artifacts preserved.',
  'actual_pin_env':expected,'reference_sha256':expected['QWEN_INDUCTOR_PIN_REFERENCE_SHA256'],'normalized_input_hashes':hashes}

def numerical_policy(nll):
 policy=json.loads((HERE/'pdl_numerical_policy.json').read_text())
 violations=[];insufficient=[]
 for row in nll['rows']:
  if min(row[side]['rounds'] for side in ('baseline','candidate'))<policy['minimum_cold_rounds_per_variant']:
   insufficient.append(row['prompt_id'])
  if abs(row['mean_delta'])>policy['maximum_absolute_prompt_mean_nll_delta']:
   violations.append({'prompt_id':row['prompt_id'],'reason':'absolute_mean_shift','value':row['mean_delta']})
  for side in ('baseline','candidate'):
   if row[side]['range']>policy['maximum_within_variant_prompt_nll_range']:
    violations.append({'prompt_id':row['prompt_id'],'reason':side+'_self_noise','value':row[side]['range']})
 return {'status':'insufficient_repeats' if insufficient else ('investigate_numerical_change' if violations else 'no_large_probe_shift_observed'),
  'policy':policy,'policy_sha256':sha((HERE/'pdl_numerical_policy.json').read_bytes()),'violations':violations,'insufficient_prompt_ids':insufficient,
  'quality_equivalence_established':False}

def main():
 p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--focused',action='store_true');args=p.parse_args()
 a,b=args.baseline.resolve(),args.candidate.resolve();require(a!=b,'same run twice')
 ma,mb=[json.loads((r/'pdl_source_manifest.json').read_text()) for r in (a,b)]
 require(ma['files']==mb['files'],'pin sources changed across pair')
 require((a/'serving/pin_reference.json').read_bytes()==(b/'serving/pin_reference.json').read_bytes(),'different pin reference')
 with tempfile.TemporaryDirectory(prefix='controlled-pdl-') as td:
  aa=normalize(a,Path(td)/'baseline'/a.name,0);bb=normalize(b,Path(td)/'candidate'/b.name,1)
  runtime=I.compare(aa[0],bb[0]);focused=None
  if args.focused:
   require((a/'focused_review.json').read_bytes()==(b/'focused_review.json').read_bytes(),'different focused frozen protocol')
   require(json.loads((a/'repeated_nll.json').read_text())['rounds']==5,'focused protocol requires five cold rounds')
   G=module('pinned_greedy',ROOT/'bench/compare_greedy_decode.py')
   S=module('pinned_serving',ROOT/'bench/compare_serving_matrix.py')
   install_pin_fi_contract(G.IDENTITY);install_pin_fi_contract(S.IDENTITY)
   ra,rb=Path(td)/'baseline'/a.name,Path(td)/'candidate'/b.name
   greedy=G.compare(G.load(ra),G.load(rb));pa,pb=S.load(ra,'prefill'),S.load(rb,'prefill')
   va,vb=pa[2]['comparison_validation'],pb[2]['comparison_validation']
   S.IDENTITY.compare(va['runtime'],vb['runtime'])
   for key in ('expected_lengths','normalized_commands'):require(va[key]==vb[key],'prefill '+key+' differs')
   require(pa[3]['run_nonce']!=pb[3]['run_nonce'],'prefill cache nonce reused')
   require(pa[3]['scenarios']==pb[3]['scenarios'],'prefill workload differs')
   focused={'greedy':greedy,'prefill_baseline':pa[2],'prefill_candidate':pb[2],
            'interpretation':'Single matched pinned timing pair, not a confidence interval; repeat reversed order for modest changes.'}
 for key in ('constructor_identity_keys',):require(aa[1][key]==bb[1][key],'different '+key)
 require(aa[1]['launched_configs'].keys()==bb[1]['launched_configs'].keys(),'different launched identity coverage')
 unique=lambda items:sorted({json.dumps(v,sort_keys=True) for v in items})
 for key in aa[1]['launched_configs']:
  require(unique(aa[1]['launched_configs'][key])==unique(bb[1]['launched_configs'][key]),'different final launch config: '+key)
 require(aa[2]['model_identities']==bb[2]['model_identities'],'different model-bound identity coverage')
 nll=N.compare(*[json.loads((r/'repeated_nll.json').read_text()) for r in (a,b)])
 pin_proof={'verified':True,'scope':'Observed guarded CachingAutotuner constructors/first launches and all actual saved-wrapper Triton source identities; external native kernels are outside this pin.',
  'reference_sha256':sha((a/'serving/pin_reference.json').read_bytes()),'constructor_identities':len(aa[1]['constructor_identity_keys']),
  'model_source_identities':len(aa[2]['model_identities']),'evidence_fields':['baseline_events','candidate_events','baseline_aot','candidate_aot']}
 runtime['generic_best_config_audit']={'scope':'Generic best_config audit not applicable to singleton-pinned compilation; authoritative pin_proof and AOT/events below.',
  'original_inductor_pinned':runtime['inductor_pinned'],'original_inductor_config_changes':runtime['inductor_config_changes']}
 runtime['inductor_pinned']=True
 runtime['inductor_config_changes']={'status':'verified_pinned_reference','pinned':True,'reference_sha256':pin_proof['reference_sha256'],'changed':{},'scope':pin_proof['scope']}
 out={'status':'controlled_configuration_verified','pin_proof':pin_proof,'runtime_changes':runtime,'baseline_events':aa[1],'candidate_events':bb[1],
  'baseline_aot':aa[2],'candidate_aot':bb[2],'normalizations':[aa[3],bb[3]],'source_manifest':ma['files'],'nll_diagnostic':nll,'numerical_probe_policy':numerical_policy(nll),'focused_timings':focused,
  'interpretation':'Same observed Triton configs and identity coverage under PDL0/1, fresh caches, fixed FI. This is NOT numerical equivalence, model quality, performance, or overlapping-consumer proof.',
  'analysis_sha256':sha(Path(__file__).read_bytes()),'compiler_audit_contract':'Empty best_config allowed ONLY with complete pinned AOT + reference + actual singleton launch checks; FI source/replay/runtime checks unchanged.'}
 args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'status':out['status'],'constructors':len(aa[1]['constructor_identity_keys']),'model_identities':len(aa[2]['model_identities'])}))
if __name__=='__main__':main()
