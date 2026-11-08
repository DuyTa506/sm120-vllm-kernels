#!/usr/bin/env python3
"""Instantiate narrow tile using measured native64x16 policy and K2048 specialization."""
import hashlib,json,re,shlex
from pathlib import Path
HERE=Path(__file__).resolve().parent;out=HERE/'rowwise_tile_control/nvcc';out.mkdir(exist_ok=True)
cache=HERE/'results/swap_native_matrix_1/jit_cache/cache';selected={}
for p in cache.glob('sm120*/kernel.cu'):
 s=p.read_text();m=re.search(r'impl<\s*(\d+), (\d+), (\d+)',s)
 if not m or int(m[2]) not in (1,3,8,16) or int(m[3])!=4096:continue
 args=[x.strip() for x in s.split('sm120_fp8_fp4_gemm_1d1d_impl<',1)[1].split('>);',1)[0].split(',')]
 assert len(args)==28 and args[6:9]==['64','16','64'] and args[12]=='16' and args[25:]==['64','4','128']
 name='tile64_m'+m[2];assert name not in selected
 args[2]='2048';args[19]='epilogue::transform::EpilogueRowChannelSwapped';args[26]='1'
 # Effective M12288 / BM64 gives192 logical tiles >=188/2, so native split policy is1.
 prefix,tail=s.split('sm120_fp8_fp4_gemm_1d1d_impl<',1);_,suffix=tail.split('>);',1)
 new=prefix+'sm120_fp8_fp4_gemm_1d1d_impl<\n        '+',\n        '.join(args)+'\n    >);'+suffix
 target=out/(name+'.cu');target.write_text(new)
 cmd=shlex.split(json.loads((p.parent/'meta.json').read_text())['command']);cmd[1]='/work/nvcc/'+target.name;cmd[cmd.index('--output-file')+1]='/work/nvcc/'+name+'.cubin'
 selected[name]={'command':cmd,'original_source':str(p.relative_to(HERE)),'original_source_sha256':hashlib.sha256(s.encode()).hexdigest(),'source_sha256':hashlib.sha256(new.encode()).hexdigest(),
 'policy':{'block_m':64,'block_n':16,'block_k':64,'stages':16,'smem':88320,'grid_ctas':188,'logical_tiles':192,'split_k':1}}
assert set(selected)=={'tile64_m1','tile64_m3','tile64_m8','tile64_m16'}
(out/'commands.json').write_text(json.dumps(selected,indent=2)+'\n');print('prepared narrow templates',len(selected))
