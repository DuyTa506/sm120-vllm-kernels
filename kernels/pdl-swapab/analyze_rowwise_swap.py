#!/usr/bin/env python3
"""Report all tiny-rowwise candidates against CUTLASS and skinny, including regressions."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bind_native_template as T
from pathlib import Path

def require(ok,message):
 if not ok:raise ValueError(message)
def timings(row):
 data=row['timings'];require(set(data)=={'cutlass','skinny','off','on'},'missing comparator')
 for values in data.values():
  require(len(values)==2,'need forward/reverse backend order')
  require(all(len(x)==3 and 0<x[1]<=x[0]<=x[2] for x in values),'invalid timing distribution')
 result={}
 for mode in ('off','on'):
  result[mode]={'median_us_by_order':[x[0] for x in data[mode]],'latency_reduction_percent':{
   baseline:[100*(1-data[mode][i][0]/data[baseline][i][0]) for i in range(2)] for baseline in ('cutlass','skinny')}}
 result['on_vs_off_reduction_percent']=[100*(1-data['on'][i][0]/data['off'][i][0]) for i in range(2)]
 return result

def analyze(root):
 for name in ('exit_code.txt','cleanup_status.txt'):require((root/name).read_text().strip()=='0','incomplete/failed '+name)
 require((root/'source_fingerprint.json').read_bytes()==(root/'source_fingerprint_after.json').read_bytes(),'source tree changed')
 require(not (root/'container_after.txt').read_text().strip() and not (root/'gpu_after.txt').read_text().strip(),'cleanup/vacancy incomplete')
 report=json.loads((root/'results.json').read_text());require(report['status']=='passed_actual_rowwise_swap','experiment did not pass operator screen')
 build=json.loads((root/'harness/build_result.json').read_text());require(report['binary_sha256']==build['extension_sha256'],'unexpected extension')
 rows=[]
 for row in report['cases']:
  require(row['skinny_pick']==[1,4] and row['captured_baseline_exact'] is True,'incorrect/missing existing comparator')
  require(row['rejected_unsupported_m']==[0,17],'tiny API did not reject unsupported shapes')
  require(set(row['input_hashes'])=={'a','sa','sb','weight'},'actual inputs not fingerprinted')
  require({x['change'] for x in row['changed_checks']}=={'actual_second_slice','zero_token','nonuniform_scales'},'missing mutation coverage')
  for change in row['changed_checks']:require(all(change['graph_eager_exact'].get(k) is True for k in ('cutlass','skinny','off','on')),'changed graph mismatch')
  for values in [row['initial_numerics']]+[c['numerics'] for c in row['changed_checks']]:
   require(set(values)=={'cutlass','skinny','off','on'},'incomplete numerical comparator')
   for label,stats in values.items():
    require(stats['relative_l2_vs_cutlass']<=0.001,'numerical screening failed')
    require(stats['relative_l2_vs_fp32']<=max(1e-6,values['cutlass']['relative_l2_vs_fp32']*1.01),'FP32 error margin failed')
  bindings={}
  for mode in ('off','on'):
   tile=re.findall(r'block_m=(\d+), block_n=(\d+), block_k=(\d+)',row['native_selection_trace'][mode])
   require(len(tile)==1,'ambiguous selected rowwise tile')
   bindings[mode]=T.bind(root,m=row['m'],n=row['n'],k=row['k'],swapped=mode=='on',
    epilogue='epilogue::transform::EpilogueRowChannel'+('Swapped' if mode=='on' else ''),tile=tuple(map(int,tile[0])))
   require(bindings[mode]['split_k']==1,'rowwise compiled kernel is not splitK1')
  rows.append({'layer':row['layer'],'m':row['m'],'n':row['n'],'k':row['k'],'authoritative_generated_template':bindings,'timing_comparison':timings(row),'raw_timings':row['timings'],'initial_numerics':row['initial_numerics'],'native_selection_trace':row['native_selection_trace']})
 require(rows,'no measured cases')
 return {'status':'validated_actual_operator_screen','cases':rows,'input_result_sha256':hashlib.sha256((root/'results.json').read_bytes()).hexdigest(),
  'interpretation':'Positive reduction is faster; negative is slower. Every orientation is retained even if slower. Forward/reverse backend-order medians are separate; no confidence interval, no vLLM throughput or model quality claim.'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=analyze(a.run);a.output.write_text(json.dumps(out,indent=2)+'\n');print(out['status'])
