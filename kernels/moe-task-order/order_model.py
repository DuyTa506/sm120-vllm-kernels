"""CPU model of native packed descriptors; models only slot permutation."""
def descriptors(tile_experts,valid_rows,gate_tiles,split_count,long_first=False):
    total=len(tile_experts)
    assert 0<=split_count<=total and gate_tiles>0 and len(valid_rows)==total
    groups=(gate_tiles+1)//2
    slots=[None]*(total+split_count*(groups-1))
    for tile,(expert,rows) in enumerate(zip(tile_experts,valid_rows)):
        assert 0<=expert<65536 and 0<rows<=128
        split=tile<split_count
        start=((total-split_count)+tile*groups if split else tile-split_count) if long_first else tile+min(tile,split_count)*(groups-1)
        chunk=2 if split else gate_tiles
        for group in range(groups if split else 1):
            begin=group*chunk
            slot=start+group
            assert slots[slot] is None,'duplicate slot'
            slots[slot]=(expert|(tile<<16),rows|(begin<<8)|(min(chunk,gate_tiles-begin)<<20))
    assert all(value is not None for value in slots),'queue hole'
    return slots
