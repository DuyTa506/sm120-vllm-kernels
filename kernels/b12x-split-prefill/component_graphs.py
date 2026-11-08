"""Separate graph-component diagnostic; never sum these as full-call latency."""
import statistics
import torch
import cuda.bindings.driver as cuda
from device_publisher import publish

def measure(candidate,x,ids,weights):
 w=candidate.w
 def quant():torch.ops._C.scaled_fp4_quant.out(x,candidate.a1,True,output=w.q,output_scale=w.sf)
 def publisher():
  publish(ids,weights,w);w.intermediate.zero_();w.output.zero_()
 def phase1():
  s=cuda.CUstream(torch.cuda.current_stream(w.output.device).cuda_stream)
  candidate.compiled[0](*candidate.args1,*candidate.ints1,s)
 def phase2():
  s=cuda.CUstream(torch.cuda.current_stream(w.output.device).cuda_stream)
  candidate.compiled[1](*candidate.args2,*candidate.ints2,s)
 calls={'quant':quant,'publisher_and_clears':publisher,'phase1':phase1,'phase2':phase2};graphs={}
 for name,fn in calls.items():
  candidate.run(x,ids,weights);torch.cuda.synchronize()
  # All kernels compiled by the earlier complete call. Standalone capture
  # reuses exact buffers; no instrumentation gets inserted into the core.
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):fn()
  graphs[name]=g
 orders=[list(calls),list(reversed(calls))];result=[]
 for order in orders:
  samples={k:[] for k in order}
  for _ in range(12):
   for name in order:
    # Prepare a valid fixed domain/intermediate every time, and prevent
    # phase2 accumulating indefinitely. Both operations are OUTSIDE timing.
    candidate.run(x,ids,weights);w.output.zero_()
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    start.record();graphs[name].replay();end.record();end.synchronize()
    samples[name].append(start.elapsed_time(end))
  result.append(dict(order=order,milliseconds=samples,medians_ms={k:statistics.median(v) for k,v in samples.items()}))
 candidate.run(x,ids,weights);torch.cuda.synchronize()
 return dict(orders=result,scope='Each sample is one standalone component graph after full-call preparation and output clear outside its event interval; publisher includes all core clears, phase2 excludes its preceding reset. Buffers/cache state differ from contiguous full-call execution. Do not sum components as a full-call result or use as isolated speedup claim.',source='Unmodified candidate phase ABI and publisher')
