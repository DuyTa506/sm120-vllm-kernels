"""Strict measurement-period verification independent of shutdown-hook delivery."""
import hashlib,importlib.util,json,sys
from pathlib import Path
SOURCE_SHA='7b5146f9b3c1e7cc09586c2087988ea8aaf9da1d86da036539a32dec4b335ec3'
RUNNER='vllm.v1.worker.gpu.model_runner'

def check_runtime(start,post,rpc,fi):
 assert rpc=={'results':[post]},'RPC/file evidence differs'
 assert start['stage']=='startup.json' and post['stage']=='post_workload.json'
 assert isinstance(post['pid'],int) and post['pid']>0 and start['pid']==post['pid']==fi['server_worker_pid'],'worker identity mismatch'
 assert start['runner']==post['runner']==RUNNER
 expected=dict(method='mtp',num_speculative_tokens=2,num_speculative_tokens_per_batch_size=None)
 assert start['speculative_config']==post['speculative_config']==expected
 assert start['source_verified'] is True and post['source_verified'] is True
 assert start['workspace_locked'] is True and post['workspace_locked'] is True
 assert start['enabled']==post['enabled'] and isinstance(post['enabled'],bool)
 assert start['resources']==post['resources'],'resource identity/lifetime changed'
 assert start['compile_factors_sha256']==post['compile_factors_sha256']
 assert start['runtime_source_hashes']==post['runtime_source_hashes'] and len(post['runtime_source_hashes'])>=4
 assert post['speculator_class']=='vllm.v1.worker.gpu.spec_decode.mtp.speculator.MTPSpeculator'
 assert post['live_graph_replays'].get('3',post['live_graph_replays'].get(3,0))>0
 groups={}
 for r in post['resources']:
  assert r['warmup_pointers_released'] is True
  key=(r['owner'],r['expert']);shapes=groups.setdefault(key,set())
  assert r['m'] not in shapes,'duplicate resource key';shapes.add(r['m'])
 if post['enabled']:
  assert groups and all(v=={1,3,4} for v in groups.values())
  owners={owner for owner,expert in groups}
  assert all(sum(owner==o for owner,expert in groups)==40 for o in owners),'not all40 target layers'
 else:assert not groups
 assert fi['import_mode']=='fixed_reference_replay_v1'
 assert fi['preload']['returned'] is True and fi['preload']['boundary']=='before_worker_load_model'
 assert fi['server_startup_boundary']=='after_compile_or_warm_up_model'
 assert fi['last_verified_stage'] in ('server_post_workload','server_shutdown')
 assert 'server_post_workload' in fi['checks'],'missing FI post boundary'
 for stage in ['server_startup','server_post_workload']:
  row=fi['checks'][stage]
  assert row['reference_entries']==row['actual_entries']==132
  assert row['missing']==row['extra']==row['changed']==[]
  assert row['entries_sha256']==fi['reference_entries_sha256']

def verify(root,helper_path):
 root=Path(root);f=root/'serving/fi_control/autotune';runtime=root/'serving/runtime'
 spec=importlib.util.spec_from_file_location('fi_post_verifier_helper',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);h=helper._helpers()
 load=lambda p:json.loads(p.read_text())
 start=load(runtime/'startup.json');post=load(runtime/'post_workload.json');rpc=load(root/'post_workload_rpc.json');fi=load(f/'verification.json')
 check_runtime(start,post,rpc,fi)
 source=(f/'source.json').read_bytes();assert hashlib.sha256(source).hexdigest()==fi['source_sha256']==SOURCE_SHA
 reference=h.tactic_entries(json.loads(source));assert len(reference)==132
 digest=h.entries_sha256(reference);assert digest==fi['reference_entries_sha256']
 for stage in ['server_startup','server_post_workload']:
  effective=h.tactic_entries(load(f/(stage+'_effective.json')))
  assert len(effective)==132 and h.entries_sha256(effective)==digest,'effective tactic keys/values drift'
 return dict(passed=True,verified_stage='explicit_server_post_workload_rpc',worker_pid=post['pid'],
  enabled=post['enabled'],source_sha256=SOURCE_SHA,entries=132,effective_entries_sha256=digest,
  shutdown_checkpoint_present='server_shutdown' in fi['checks'],inductor_pinned=False)
if __name__=='__main__':print(json.dumps(verify(*sys.argv[1:]),indent=2))
