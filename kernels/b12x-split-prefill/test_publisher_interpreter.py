"""Run actual Triton publisher under CPU interpreter; not GPU codegen proof."""
import os
os.environ['TRITON_INTERPRET']='1'
import torch
from types import SimpleNamespace
from device_publisher import publish
from domain import scale_offset
from numerical_reference import validate_domain

def main():
 torch.set_num_threads(2);m=129;ct=(m*8+127)//128+255;rows=ct*128
 alloc=lambda n,d:torch.empty(n,dtype=d)
 w=SimpleNamespace(m=m,capacity_tiles=ct,q=torch.arange(m*1024,dtype=torch.int64).to(torch.uint8).reshape(m,1024),sf_bytes=torch.arange(((m+127)//128)*128*128,dtype=torch.int64).to(torch.uint8),counts=alloc(256,torch.int32),write_rows=alloc(256,torch.int32),errors=alloc(1,torch.int32),tile_bases=alloc(257,torch.int32),packed_a=alloc(rows*1024,torch.uint8),scale_a=alloc(rows*128,torch.uint8),token_map=alloc(rows,torch.int32),route_slots=alloc(rows,torch.int32),token_weights=alloc(rows,torch.float32),task_expert=alloc(ct*4,torch.int32),task_rows=alloc(ct*4,torch.int32))
 for change in range(4):
  ids=((torch.arange(m*8).reshape(m,8)+change*19)%23).to(torch.int32)
  if change==0:ids=torch.arange(8,dtype=torch.int32).expand(m,8).contiguous()
  if change==1:ids[::2,::2]=-1
  if change==3:ids.fill_(-1)
  weights=torch.arange(m*8,dtype=torch.float32).reshape(m,8)/137
  w.q.copy_((w.q.to(torch.int32)+7).to(torch.uint8));w.sf_bytes.copy_((w.sf_bytes.to(torch.int32)+11).to(torch.uint8))
  publish(ids,weights,w)
  assert validate_domain(w,ids,weights)["passed"]
  assert w.errors.item()==0
  expected=torch.bincount(ids[ids>=0].long(),minlength=256);assert torch.equal(w.counts,expected)
  seen=[]
  for tile in range(int(w.tile_bases[-1])):
   e=int(w.task_expert[tile*4]);valid=int(w.task_rows[tile*4]);assert 0<valid<=128
   for row in range(tile*128,(tile+1)*128):
    token=int(w.token_map[row]);slot=int(w.route_slots[row])
    if row%128>=valid:assert token==slot==-1;assert not w.packed_a[row*1024:(row+1)*1024].any();continue
    assert int(ids[token,slot])==e;seen.append((token,slot))
    assert w.token_weights[row]==weights[token,slot]
    assert torch.equal(w.packed_a[row*1024:(row+1)*1024],w.q[token])
    for block in range(128):assert w.scale_a[scale_offset(row,block,2048)]==w.sf_bytes[scale_offset(token,block,2048)]
  assert sorted(seen)==[(t,s) for t in range(m) for s in range(8) if ids[t,s]>=0]
 ids[0,0]=256;ids[0,1]=7;ids[0,2]=7;publish(ids,weights,w);assert w.errors.item()>=2
 assert not torch.cuda.is_initialized()
 print('actual Triton publisher CPU interpreter: changed routes/input scales, invalid sentinel, duplicate/error flags and byte domain passed; no GPU/codegen proof')
if __name__=='__main__':main()
