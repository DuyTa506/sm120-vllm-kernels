"""Reject successful CLI exit with partial requests or truncated outputs."""
import json,sys
from pathlib import Path
EXPECTED={'decode_c1':(8,128,128),'decode_c4':(16,128,128),'prefill_2048':(8,2048,1)}
def check(data,n,inputs,outputs):
 assert data['completed']==n and data['failed']==0
 assert data['num_prompts']==n
 assert data['total_input_tokens']==n*inputs and data['total_output_tokens']==n*outputs
 assert data['input_lens']==[inputs]*n and data['output_lens']==[outputs]*n
 assert not any(data['errors'])
if __name__=='__main__':
 root=Path(sys.argv[1]);report={}
 for task,args in EXPECTED.items():
  data=json.loads((root/(task+'.json')).read_text());check(data,*args)
  report[task]=dict(passed=True,completed=data['completed'],output_tokens=data['total_output_tokens'])
 print(json.dumps(report))
