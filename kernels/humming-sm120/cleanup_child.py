"""Terminate only the exact tagged micro process, then prove its absence."""
import json,os,signal,sys,time
marker=sys.argv[1]+'/screen.json'
def owned():
    hits=[]
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid)==os.getpid():continue
        try:argv=open('/proc/'+pid+'/cmdline','rb').read().decode().split('\0')
        except (OSError,UnicodeError):continue
        if any('/work/kernels/humming-sm120/'+name in argv for name in ('micro.py','numerical.py')) and marker in argv:hits.append(int(pid))
    return hits
initial=owned()
for pid in initial:
    try:os.kill(pid,signal.SIGTERM)
    except ProcessLookupError:pass
for _ in range(50):
    if not owned():break
    time.sleep(.1)
for pid in owned():
    try:os.kill(pid,signal.SIGKILL)
    except ProcessLookupError:pass
for _ in range(20):
    if not owned():break
    time.sleep(.1)
remaining=owned();print(json.dumps(dict(initial=initial,remaining=remaining)))
assert not remaining
