"""GPU count/prefix/pack publisher. No host routing or data-dependent allocation.

Experimental reference implementation; launch/clearing overhead is intentional
and must be included in any complete-call timing. Not GPU-validated yet.
"""
import triton
import triton.language as tl

@triton.jit
def _count(Ids, Counts, Errors, TOPK:tl.constexpr, E:tl.constexpr):
    t=tl.program_id(0);j=tl.arange(0,TOPK);e=tl.load(Ids+t*TOPK+j)
    valid=(e>=0)&(e<E)
    duplicates=tl.sum(((e[:,None]==e[None,:]) & (j[:,None]<j[None,:]) & valid[:,None]).to(tl.int32))
    bad=tl.sum(((e < -1)|(e>=E)).to(tl.int32))+duplicates
    if bad>0:tl.atomic_add(Errors,bad)
    tl.atomic_add(Counts+tl.where(valid,e,0),1,mask=valid)

@triton.jit
def _prefix(Counts,Bases,E:tl.constexpr):
    e=tl.arange(0,E);counts=tl.load(Counts+e);tiles=(counts+127)//128
    ends=tl.cumsum(tiles,0);tl.store(Bases+e,ends-tiles)
    tl.store(Bases+E,tl.sum(tiles,0))

@triton.jit
def _metadata(Counts,Bases,TaskExpert,TaskRows,E:tl.constexpr,NT:tl.constexpr):
    tile=tl.program_id(0);e=tl.arange(0,E);beg=tl.load(Bases+e);end=tl.load(Bases+e+1)
    owned=(tile>=beg)&(tile<end)
    expert=tl.max(tl.where(owned,e,-1),0)
    rows=tl.sum(tl.where(owned,tl.minimum(128,tl.load(Counts+e)-(tile-beg)*128),0),0)
    n=tl.arange(0,NT);tl.store(TaskExpert+tile*NT+n,expert);tl.store(TaskRows+tile*NT+n,rows)

@triton.jit
def _pack(Ids,Weights,Q,SF,Bases,WriteRows,PA,SA,TokenMap,RouteSlots,TokenWeights,
          TOPK:tl.constexpr,E:tl.constexpr,K:tl.constexpr):
    route=tl.program_id(0);token=route//TOPK;slot=route%TOPK;e=tl.load(Ids+route)
    if (e>=0)&(e<E):
        rank=tl.atomic_add(WriteRows+e,1)
        row=tl.load(Bases+e)*128+rank
        tl.store(TokenMap+row,token);tl.store(RouteSlots+row,slot)
        tl.store(TokenWeights+row,tl.load(Weights+route))
        b=tl.arange(0,K//2);tl.store(PA+row*(K//2)+b,tl.load(Q+token*(K//2)+b))
        c=tl.arange(0,K//16)
        source=(token//128)*(K//64)*512+(c//4)*512+(token%32)*16+((token%128)//32)*4+c%4
        target=(row//128)*(K//64)*512+(c//4)*512+(row%32)*16+((row%128)//32)*4+c%4
        tl.store(SA+target,tl.load(SF+source))

def publish(ids,weights,w):
    # Explicit fixed-capacity clears avoid stale tail bytes/metadata on replay.
    w.counts.zero_();w.write_rows.zero_();w.errors.zero_()
    w.packed_a.zero_();w.scale_a.zero_();w.token_map.fill_(-1)
    w.route_slots.fill_(-1);w.token_weights.zero_()
    _count[(w.m,)](ids,w.counts,w.errors,TOPK=8,E=256,num_warps=4)
    _prefix[(1,)](w.counts,w.tile_bases,E=256,num_warps=4)
    _metadata[(w.capacity_tiles,)](w.counts,w.tile_bases,w.task_expert,w.task_rows,E=256,NT=4,num_warps=4)
    _pack[(w.m*8,)](ids,weights,w.q,w.sf_bytes,w.tile_bases,w.write_rows,w.packed_a,w.scale_a,w.token_map,w.route_slots,w.token_weights,TOPK=8,E=256,K=2048,num_warps=4)
