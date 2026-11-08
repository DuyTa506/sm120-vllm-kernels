"""Signal only a verified API listener through pidfd; never the nsys PID1 tree."""
import argparse,hashlib,json,os,signal,time
from pathlib import Path

def listener_inodes(port):
    result=set()
    for name in ('tcp','tcp6'):
        for line in Path('/proc/net',name).read_text().splitlines()[1:]:
            fields=line.split()
            if fields[3]=='0A' and int(fields[1].rsplit(':',1)[1],16)==port:result.add(fields[9])
    return result

def identity(pid,port):
    root=Path('/proc')/str(pid)
    stat=root.joinpath('stat').read_text().rsplit(')',1)[1].split()
    raw=root.joinpath('cmdline').read_bytes();argv=[x.decode(errors='replace') for x in raw.split(b'\0') if x]
    exe=os.readlink(root/'exe')
    assert pid>1 and Path(exe).name.startswith('python'), 'Not a non-PID1 Python API process'
    cli=any(Path(word).name=='vllm' and index+1<len(argv) and argv[index+1]=='serve' for index,word in enumerate(argv))
    titled=bool(argv) and argv[0]=='VLLM::APIServer'
    assert cli or titled, 'Process command does not identify vLLM API'
    owned=set()
    for fd in (root/'fd').iterdir():
        try:target=os.readlink(fd)
        except FileNotFoundError:continue
        if target.startswith('socket:['):owned.add(target[8:-1])
    matched=owned & listener_inodes(port)
    assert matched, 'Process does not own the expected listening socket'
    return dict(pid=pid,start_ticks=stat[19],exe=exe,cmdline=argv,cmdline_sha256=hashlib.sha256(raw).hexdigest(),port=port,listener_inodes=sorted(matched))

def discover(port):
    found=[]
    for root in Path('/proc').iterdir():
        if not root.name.isdigit():continue
        try:record=identity(int(root.name),port)
        except (OSError,AssertionError):continue
        found.append(record)
    assert len(found)==1, f'Expected exactly one owned API listener, found {len(found)}'
    return found[0]

def signal_api(record):
    # pidfd binds a process lifetime; a recycled numeric PID cannot receive this signal.
    fd=os.pidfd_open(record['pid'],0)
    try:
        current=identity(record['pid'],record['port'])
        for key in ('pid','start_ticks','exe','cmdline_sha256','listener_inodes'):
            assert current[key]==record[key], 'API identity changed: '+key
        signal.pidfd_send_signal(fd,signal.SIGTERM,None,0)
    finally:os.close(fd)
    return dict(action='SIGTERM_verified_API_pidfd',identity=record,signalled_at=time.time(),nsys_pid1_signalled=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['discover','signal']);p.add_argument('record',type=Path);p.add_argument('--port',type=int,default=8000);a=p.parse_args()
    if a.mode=='discover':
        assert not a.record.exists();result=discover(a.port);a.record.write_text(json.dumps(result,indent=2)+'\n')
    else:result=signal_api(json.loads(a.record.read_text()))
    print(json.dumps(result))
if __name__=='__main__':main()
