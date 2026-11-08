"""Actual repository text, serial cold4096 prefill and repeated identical-token NLL."""
import argparse,hashlib,json,math,os,time,urllib.request
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--trace-only',action='store_true');p.add_argument('--prompt',type=Path,required=True);a=p.parse_args()
    url=os.environ['BENCH_URL'];text=a.prompt.read_text()
    def post(path,body):
        req=urllib.request.Request(url+path,json.dumps(body).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=600) as response:
            raw=response.read();return json.loads(raw) if raw else None
    def tokens(label):
        ids=post('/tokenize',{'model':'bench','prompt':label+'\n'+text*8})['tokens'][:4096]
        assert len(ids)==4096;return ids
    def request(ids,nll=False):
        for attempt in range(20):
            cleared=post('/reset_prefix_cache',{})
            assert isinstance(cleared,dict) and type(cleared.get('success')) is bool
            if cleared['success']:break
            time.sleep(0.05)
        else:raise RuntimeError('Prefix-cache reset was rejected; cold measurement invalid')
        body={'model':'bench','prompt':ids,'max_tokens':1,'temperature':0,'ignore_eos':True}
        if nll:body['prompt_logprobs']=0
        start=time.perf_counter();response=post('/v1/completions',body);elapsed=time.perf_counter()-start
        assert response['usage']['prompt_tokens']==4096 and response['usage']['completion_tokens']==1
        return response,elapsed
    if a.trace_only:
        ids=tokens('[task-order-native-live-trace]');response,elapsed=request(ids)
        a.output.write_text(json.dumps({'prompt_tokens':4096,'request_seconds':elapsed,'scope':'trace only, excluded from timing screen'})+'\n');return
    measured=[tokens(f'[task-order-prefill-real-{i}]') for i in range(8)]
    warm=[tokens(f'[task-order-prefill-warm-{i}]') for i in range(2)]
    for ids in warm:request(ids)
    rows=[]
    for ids in measured:
        response,elapsed=request(ids);rows.append(dict(input_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),request_seconds=elapsed,completion=response['choices'][0]['text']))
    nll_ids=tokens('[task-order-cold-quality-identical-input]');nll=[]
    for repeat in range(4):
        response,_=request(nll_ids,True);raw=response['choices'][0]['prompt_logprobs']
        assert len(raw)==len(nll_ids)
        values=[entry[str(nll_ids[index])]['logprob'] for index,entry in enumerate(raw[1:],1)]
        assert len(values)==4095 and all(math.isfinite(x) for x in values)
        nll.append(dict(repeat=repeat,nll=-sum(values)/len(values),logprobs=values))
    a.output.write_text(json.dumps(dict(prompt_source_sha256=hashlib.sha256(text.encode()).hexdigest(),prefill=rows,nll_input_ids_sha256=hashlib.sha256(json.dumps(nll_ids).encode()).hexdigest(),nll=nll,cold_policy='explicit reset_prefix_cache before every request',timing_scope='client nonstream request latency for exact4096 input/1 output; not a streaming TTFT measurement'),indent=2)+'\n')
if __name__=='__main__':main()
