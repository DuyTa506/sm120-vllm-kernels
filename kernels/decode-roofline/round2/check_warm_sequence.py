"""Check completed warm transitions; does not substitute for kernel activity traces."""
import argparse
import json
from pathlib import Path


def check(paths):
    rows=[]
    for path in paths:
        assert (path/'exit_code.txt').read_text().strip()=='0',path
        record=json.loads((path/'serving/skinny_runtime/runtime.json').read_text())
        assert record['source_verified'] is True
        for c in (1,4):
            data=json.loads((path/f'serving/c{c}.json').read_text())
            assert data['completed']==32 and data['failed']==0,(path,c)
        rows.append(record)
    assert [r['profile'] for r in rows][::2]==['off','off']
    assert rows[1]['profile']==rows[3]['profile']!='off'
    assert len({r['compile_cache_root'] for r in rows})==1,'test did not reuse cache root'
    hashes=[r['compile_factor_sha256'] for r in rows]
    assert hashes[0]==hashes[2] and hashes[1]==hashes[3] and hashes[0]!=hashes[1], 'unexpected factor identities'
    return dict(config_and_serving_pass=True, profiles=[r['profile'] for r in rows],
                compile_factor_sha256=hashes, captured_kernel_activity_verified=False,
                limitation='trace actual captured kernels before reporting warm-cache activation correctness')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('paths',type=Path,nargs=4);args=parser.parse_args()
    print(json.dumps(check(args.paths),indent=2))
