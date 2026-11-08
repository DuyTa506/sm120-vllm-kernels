"""Synthetic artifact gate checks; never changes the preserved serving evidence."""
import json,shutil,tempfile
from pathlib import Path
from verify_post_workload import verify
here=Path(__file__).resolve().parent
repo=here.parents[2]
plugin=repo/'kernels/decode-route-capture/qwen_fi_control_plugin.py'
source_sha='7b5146f9b3c1e7cc09586c2087988ea8aaf9da1d86da036539a32dec4b335ec3'
original=here/'results/dg_serving_off2/serving'
with tempfile.TemporaryDirectory(prefix='dg-synthetic-checkpoint-') as temp:
    serving=Path(temp);fi=serving/'fi_control/autotune';fi.mkdir(parents=True)
    runtime=serving/'dg_runtime';runtime.mkdir()
    for name in ('source.json','verification.json','server_startup_effective.json'):
        shutil.copyfile(original/'fi_control/autotune'/name,fi/name)
    report=json.loads((fi/'verification.json').read_text());report['server_checkpoint']='warmup_verified';(fi/'verification.json').write_text(json.dumps(report))
    try:verify(serving,plugin,source_sha)
    except (AssertionError,FileNotFoundError):pass
    else:raise AssertionError('missing explicit workload checkpoint accepted')
    report['server_post_workload_boundary']='explicit_collective_rpc_after_tasks';(fi/'verification.json').write_text(json.dumps(report))
    shutil.copyfile(fi/'server_startup_effective.json',fi/'server_post_workload_effective.json')
    dg=json.loads((original/'dg_runtime/startup.json').read_text());dg['live_full_graph_replays']={'48':13};(runtime/'post_workload.json').write_text(json.dumps(dg))
    result=verify(serving,plugin,source_sha);assert result['verified_stage']=='server_post_workload' and result['shutdown_proven'] is False
    (fi/'server_post_workload_effective.json').write_text('{}')
    try:verify(serving,plugin,source_sha)
    except (AssertionError,ValueError):pass
    else:raise AssertionError('changed effective FI table accepted')
print('post-workload gate rejects missing checkpoint and FI drift; never claims shutdown')
