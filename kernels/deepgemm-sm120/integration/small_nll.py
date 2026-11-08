"""Cold repository real-text NLL at DG-eligible prefill lengths, same requests in both controls."""
import hashlib,json,math,os,sys,urllib.request
from pathlib import Path
url=os.environ['BENCH_URL']
text=Path(sys.argv[2]).read_text()
def post(path,body):
    request=urllib.request.Request(url+path,json.dumps(body).encode(),{'Content-Type':'application/json'})
    return json.loads(urllib.request.urlopen(request,timeout=600).read())
rows=[]
for length in (48,64,96,128):
    prefix=f'[dg-rowwise-quality-{length}]\n'
    ids=post('/tokenize',{'model':'bench','prompt':prefix+text})['tokens'][:length]
    assert len(ids)==length
    response=post('/v1/completions',{'model':'bench','prompt':ids,'max_tokens':1,'temperature':0,'prompt_logprobs':0})
    raw=response['choices'][0]['prompt_logprobs']
    values=[list(x.values())[0]['logprob'] for x in raw[1:] if x]
    assert len(values)==length-1 and all(math.isfinite(v) for v in values)
    rows.append({'tokens':length,'input_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                 'nll':-sum(values)/len(values),'logprobs':values})
Path(sys.argv[1]).write_text(json.dumps({'prompt_source_sha256':hashlib.sha256(text.encode()).hexdigest(),'cases':rows},indent=2)+'\n')
