"""Fixed minimal counter protocol; constructing argv never touches CUDA."""
METRICS=('gpu__time_duration.sum','dram__bytes.sum.per_second','lts__throughput.avg.pct_of_peak_sustained_elapsed','lts__t_sector_hit_rate.pct','sm__throughput.avg.pct_of_peak_sustained_elapsed','sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed')
RANGE='scalar_L19_M1024'
KERNEL='regex:.*gatedMoEGatedDynamicKernel.*'
def command(here,output):
 return [str(here/'tools/linux-desktop-glibc_2_11_3-x64/ncu'),'--replay-mode','application','--app-replay-mode','strict','--cache-control','none','--clock-control','none','--nvtx','--nvtx-include',RANGE+'/','--kernel-name',KERNEL,'--launch-count','1','--metrics',','.join(METRICS),'--section-folder',str(here/'tools/sections'),'--export',str(output/'counters'),'--page','raw','--csv','/usr/bin/python3',str(here/'baseline.py'),'--output',str(output)]
def classify(text,code):
 if 'ERR_NVGPUCTRPERM' in text:return 'counter_permission_unavailable'
 if any(s in text.lower() for s in ['invalid metric','unknown metric','metric not found']):return 'unsupported_metric'
 if code:return 'profiler_or_application_failed'
 if 'No kernels were profiled' in text:return 'selection_failed'
 return 'requires_report_validation'
