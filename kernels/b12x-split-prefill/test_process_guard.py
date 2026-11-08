import json,os,subprocess,sys,tempfile
from pathlib import Path
here=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as td:
 child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
 try:
  tick=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
  marker=Path(td)/'process.json';record=dict(pid=child.pid,pgid=child.pid,start_ticks=tick,state='running')
  # Concurrent reaper matches real launcher semantics, avoiding a zombie held
  # by this fixture parent while the cleanup subprocess checks /proc absence.
  import threading
  reaper=threading.Thread(target=child.wait);reaper.start()
  marker.write_text(json.dumps(record))
  run=subprocess.run([sys.executable,str(here/'process_guard.py'),'stop',td],capture_output=True,text=True)
  assert run.returncode==0,(run.stdout,run.stderr)
  result=json.loads(run.stdout);assert result['remaining_group_pids']==[]
  reaper.join(timeout=2);assert child.poll() is not None
 finally:
  if child.poll() is None:child.kill();child.wait()
print('owned process-group cleanup CPU fixture passed')
