"""Reference M128 expert-major domain, independent of any GEMM scheduler.

CPU setup/validation only. A device publisher must match this semantic mapping,
not its row order; expert-local order may differ under atomic placement.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class Domain:
    counts: tuple
    tile_bases: tuple
    token_map: tuple
    route_slots: tuple
    task_expert: tuple
    task_valid_rows: tuple
    source_tile_m: int = 128
    intermediate_tiles: int = 4

def publish(ids, experts=256, intermediate=512):
    if intermediate % 128: raise ValueError('intermediate must be divisible by128')
    buckets=[[] for _ in range(experts)]
    for token,row in enumerate(ids):
        if len(set(row))!=len(row): raise ValueError('duplicate expert per token')
        for top,expert in enumerate(row):
            if not 0<=expert<experts: raise ValueError('expert out of bounds')
            buckets[expert].append((token,top))
    bases=[0]
    for rows in buckets:bases.append(bases[-1]+(len(rows)+127)//128)
    token_map=[-1]*(bases[-1]*128);slots=[-1]*len(token_map)
    nt=intermediate//128;te=[-1]*(bases[-1]*nt);vr=[0]*len(te)
    for expert,rows in enumerate(buckets):
        for row,(token,top) in enumerate(rows):
            pos=bases[expert]*128+row;token_map[pos]=token;slots[pos]=top
        for tile in range(bases[expert],bases[expert+1]):
            valid=min(128,len(rows)-(tile-bases[expert])*128)
            for n in range(nt):te[tile*nt+n]=expert;vr[tile*nt+n]=valid
    return Domain(tuple(map(len,buckets)),tuple(bases),tuple(token_map),tuple(slots),tuple(te),tuple(vr),intermediate_tiles=nt)

def intermediate_bytes(rows_capacity,intermediate=512):
    if rows_capacity%128 or intermediate%128:raise ValueError('invalid domain geometry')
    return rows_capacity*(intermediate//128)*18*4

def scale_offset(row,block,k):
    if k%64:raise ValueError('K must be divisible by64')
    return (row//128)*(k//64)*512+(block//4)*512+(row%32)*16+((row%128)//32)*4+block%4
