#!/usr/bin/env python3
"""Compare completed native off/on policy matrices; never assert model applicability."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bind_native_template as T

def sha(data):return hashlib.sha256(data).hexdigest()
def require(ok,message):
 if not ok:raise ValueError(message)
def load(root):
 require((root/'exit_code.txt').read_text().strip()=='0','failed native run')
 require((root/'cleanup_status.txt').read_text().strip()=='0','failed cleanup')
 require(not (root/'container_after.txt').read_text().strip(),'owned container remains')
 require(not (root/'gpu_after.txt').read_text().strip(),'GPU not empty after run')
 require((root/'source_fingerprint.json').read_bytes()==(root/'source_fingerprint_after.json').read_bytes(),'native sources changed')
 doc=json.loads((root/'results.json').read_text());require(doc['status']=='complete_native_policy_micro','partial native matrix')
 cases={}
 for row in doc['cases']:
  key=(row['m'],row['n'],row['k']);require(key not in cases,'duplicate shape')
  require(row['on_vs_off_error']['relative_rms']==0 and row['on_vs_off_error']['max_error_over_rms']==0,'paired outputs differ')
  for mode in ('off','on'):
   for checks in ('reference_error','changed_input_replay'):
    e=row[checks][mode];require(e['finite'] and e['relative_rms']<=0.01 and e['max_error_over_rms']<=0.05,'numerical gate failed')
   marker=f'SWAP_AB_CONTROL_V1 m={key[0]} n={key[1]} k={key[2]} eligible=1 selected={int(mode=="on")}'
   require(marker in row['native_selection_trace'][mode],'selection proof missing')
   require(len(row['samples_us'][mode])==doc['plan']['iterations'],'sample count mismatch')
   require(statistics.median(row['samples_us'][mode])==row['median_us'][mode],'incorrect median')
  cases[key]=row
 return doc,cases

def main():
 p=argparse.ArgumentParser();p.add_argument('baseline_order',type=Path);p.add_argument('reverse_order',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 paths=[a.baseline_order,a.reverse_order];loaded=[load(r) for r in paths];x,y=[z[0] for z in loaded]
 for k in ('binary_sha256','script_sha256','helper_sha256','gpu','source_manifest'):require(x[k]==y[k],'different '+k)
 for name in ('sources.sha256','source_fingerprint.json','image_id.txt'):require((paths[0]/name).read_bytes()==(paths[1]/name).read_bytes(),'different '+name)
 require(x['plan']['ms']==list(reversed(y['plan']['ms'])) and x['plan']['shapes_nk']==list(reversed(y['plan']['shapes_nk'])),'cell order not reversed')
 for k in ('iterations','cold_mib','scope'):require(x['plan'][k]==y['plan'][k],'different plan '+k)
 require(loaded[0][1].keys()==loaded[1][1].keys(),'shape coverage differs')
 rows=[]
 for key in sorted(loaded[0][1]):
  per=[];configs=[]
  for path,(_,cases) in zip(paths,loaded):
   c=cases[key];delta=[100*(1-on/off) for off,on in zip(c['samples_us']['off'],c['samples_us']['on'],strict=True)]
   config={mode:re.findall(r'block_m=\d+, block_n=\d+, block_k=\d+|num_stages=\d+|split_k=\d+',trace) for mode,trace in c['native_selection_trace'].items()};configs.append(config)
   bindings={}
   for mode in ('off','on'):
    tile=re.findall(r'block_m=(\d+), block_n=(\d+), block_k=(\d+)',c['native_selection_trace'][mode])
    require(len(tile)==1,'ambiguous tile trace')
    bindings[mode]=T.bind(path,m=key[0],n=key[1],k=key[2],swapped=mode=='on',epilogue='epilogue::transform::EpilogueIdentity',tile=tuple(map(int,tile[0])))
   per.append({'run':path.name,'median_us':c['median_us'],'median_latency_reduction_percent':100*(1-c['median_us']['on']/c['median_us']['off']),
    'paired_sample_reduction_percent':{'min':min(delta),'median':statistics.median(delta),'max':max(delta)},'observed_pre_finalization_config':config,'authoritative_generated_template':bindings,'weight_pool_bytes':c['weight_pool_bytes'],
    'reference_error':c['reference_error'],'changed_input_replay':c['changed_input_replay'],'on_vs_off_error':c['on_vs_off_error']})
  require(configs[0]==configs[1],'native policy configs changed across repeats')
  rows.append({'m':key[0],'n':key[1],'k':key[2],'runs':per})
 report={'status':'verified_native_policy_repeat','rows':rows,'sources':{str(r):{'result_sha256':sha((r/'results.json').read_bytes()),'source_fingerprint_sha256':sha((r/'source_fingerprint.json').read_bytes())} for r in paths},
  'split_k_correction':'DG_PRINT_CONFIGS logs get_best_sm120_config before caller finalizes split-K. Original traces preserved. Generated template parameters are authoritative: N2048/K4096 off/on splitK4; N12288/K2048 off/on splitK1.',
  'binary_sha256':x['binary_sha256'],'script_sha256':x['script_sha256'],'analysis_sha256':sha(Path(__file__).read_bytes()),
  'interpretation':'Native UE8M0 FP8 policy off/on with same operands within each cell; policy includes tile/stage changes. Existing auto-swap already enables this path. Fresh reversed-cell RNG consumption differs, so matched cells across runs do not claim identical tensors. No vLLM/NVFP4 integration or model speedup established. Per-run reductions retained; no pooled headline or confidence interval.'}
 a.output.write_text(json.dumps(report,indent=2)+'\n')
 for row in rows:print(row['m'],row['n'],row['k'],*[round(v['median_latency_reduction_percent'],2) for v in row['runs']])
if __name__=='__main__':main()
