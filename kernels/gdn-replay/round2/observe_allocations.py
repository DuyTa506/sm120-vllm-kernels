"""Host-only diagnostic trajectory; raw gauges, no inferred eviction claims."""
import argparse,json,signal,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
alive=True
def stop(*args):
    global alive
    alive=False
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
keys=('vllm:kv_cache_usage_perc','vllm:gpu_cache_usage_perc','vllm:num_preemptions_total','vllm:num_requests_running','vllm:num_requests_waiting')
with a.output.open('x') as f:
    while alive:
        row=dict(monotonic_s=time.monotonic(),unix_s=time.time())
        try:
            text=urllib.request.urlopen(a.url+'/metrics',timeout=3).read().decode()
            row['metrics']=[line for line in text.splitlines() if line.startswith(keys)]
        except Exception as e:row['error']=repr(e)
        f.write(json.dumps(row)+'\n');f.flush()
        time.sleep(1)
