"""Verify a worker-alive post-workload checkpoint, separately from shutdown."""
import argparse,hashlib,importlib.util,json
from pathlib import Path

def verify(serving, plugin_path, source_sha):
    spec=importlib.util.spec_from_file_location('fi_explicit_checkpoint_verifier',plugin_path)
    fi=importlib.util.module_from_spec(spec);spec.loader.exec_module(fi)
    startup=fi.verify_artifacts(serving/'fi_control',source_sha,require_shutdown=False)
    root=serving/'fi_control/autotune'
    report=json.loads((root/'verification.json').read_text())
    assert report.get('server_post_workload_boundary')=='explicit_collective_rpc_after_tasks'
    helpers=fi._helpers()
    entries=helpers.tactic_entries(json.loads((root/'server_post_workload_effective.json').read_text()))
    effective_sha=helpers.entries_sha256(entries)
    assert effective_sha==report['reference_entries_sha256']
    runtime=json.loads((serving/'dg_runtime/post_workload.json').read_text())
    assert runtime['source_verified'] and runtime['deepgemm_pdl'] is False
    assert runtime['matched_qkvz_modules']==30
    assert runtime['enabled_qkvz_modules']==(30 if runtime['flag'] else 0)
    assert int(runtime['live_full_graph_replays'].get('48',0))>0
    return dict(verified_stage='server_post_workload',boundary=report['server_post_workload_boundary'],
                effective_entries_sha256=effective_sha,entries=len(entries),deepgemm_pdl=False,
                enabled_qkvz_modules=runtime['enabled_qkvz_modules'],shutdown_proven=False,
                startup_source_sha256=startup['source_sha256'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('serving',type=Path);p.add_argument('fi_plugin',type=Path);p.add_argument('source_sha');a=p.parse_args()
    print(json.dumps(verify(a.serving,a.fi_plugin,a.source_sha),indent=2))
