"""CPU arithmetic/layout/route contract checks; never imports downloaded code."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def scale_offset(row,quad,kq):
    return ((row>>7)*kq+quad)*512+(row&31)*16+((row&127)>>5)*4

def geometry(h,i,topk):
    assert h%256==0 and i%128==0 and 8*topk<=128
    return dict(H=h,I=i,TOPK=topk,MAXT=8,HB=h//16,IB=i//16,HQ=h//64,IQ=i//64,
                fc1_register_quads=h//256,fc2_register_quads=i//128,
                fc1_tiles=i//16,fc2_tiles=h//128,
                max_fc1_dynamic_smem=8*(h//8*8+h//16*4),
                partial_bytes=8*topk*8*h*4)

def check():
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'sources').iterdir())}
    src=(ROOT/'sources/qf_moe.cu').read_text()
    for needle in ['H = 2560, I = 640, TOPK = 10','uint4 wv[10][2]','uint4 wv[5][2]','__dp4a','static at::Tensor part_out, done','__syncthreads_or(tid < pos && s_ids[tid] == e)']:
        assert needle in src,needle
    # Exhaustive FP4-code products: signed x2 magnitudes yield exact quarter products.
    values=[0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6]
    ints=[int(x*2) for x in values]
    for a in range(16):
        for b in range(16):assert ints[a]*ints[b]*.25==values[a]*values[b]
    assert 16*12*12 < 2**31
    # Packed four-scale offsets cover physical buffer exactly, for both weight matrices.
    target=geometry(2048,512,8)
    for rows,kq in [(1024,32),(2048,8)]:
        offsets=[scale_offset(r,q,kq)+b for r in range(rows) for q in range(kq) for b in range(4)]
        assert len(set(offsets))==rows*kq*4 and min(offsets)==0 and max(offsets)==len(offsets)-1
    # First owner handles repeated experts across tokens; duplicate within a token forbidden.
    routes=[list(range(8)),list(range(4,12)),list(range(8))]
    flat=sum(routes,[]);owners={e:flat.index(e) for e in flat}
    for t,row in enumerate(routes):
        assert len(set(row))==8
        for e in row:assert flat[owners[e]]==e
    all_same=[list(range(8)) for _ in range(8)]
    disjoint=[list(range(t*8,(t+1)*8)) for t in range(8)]
    return dict(cpu_only=True,downloaded_code_executed=False,source_sha256=hashes,
                original=geometry(2560,640,10),target=target,
                exact_fp4_product_pairs=256,scale_layout_bijection=True,
                route_fixtures_unique_experts=[len(set(sum(x,[]))) for x in [routes,all_same,disjoint]],
                note='Does not prove CUDA quantization parity, actual layout identity, races, or speed.')

if __name__=='__main__':
    print(json.dumps(check(),indent=2))
