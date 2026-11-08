import argparse,importlib.util,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('serving',type=Path);p.add_argument('plugin',type=Path);p.add_argument('source_sha');a=p.parse_args()
spec=importlib.util.spec_from_file_location('fi_task_order_check',a.plugin);fi=importlib.util.module_from_spec(spec);spec.loader.exec_module(fi)
startup=fi.verify_artifacts(a.serving/'fi_control',a.source_sha,require_shutdown=False)
root=a.serving/'fi_control/autotune';report=json.loads((root/'verification.json').read_text())
assert report['server_post_workload_boundary']=='explicit_collective_rpc_after_tasks'
helpers=fi._helpers();entries=helpers.tactic_entries(json.loads((root/'server_post_workload_effective.json').read_text()))
assert helpers.entries_sha256(entries)==report['reference_entries_sha256']
runtime=json.loads((a.serving/'task_order/post_workload.json').read_text())
assert runtime['queues'] and any(q['nontrivial'] for q in runtime['queues'])
print(json.dumps(dict(stage='server_post_workload',profile=runtime['profile'],entries=len(entries),queues=len(runtime['queues']),shutdown_proven=False,source_sha256=startup['source_sha256'])))
