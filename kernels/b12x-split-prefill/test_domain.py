from domain import publish,intermediate_bytes,scale_offset
# Empty experts, exact128, multi-tile129, asymmetric tails, duplicated router IDs.
for m in (0,1,63,64,65,127,128,129,1024,4096):
 ids=[[i%3,(i+1)%3] for i in range(m)]
 d=publish(ids,experts=4)
 seen=[]
 for tile in range(d.tile_bases[-1]):
  e=d.task_expert[tile*4];valid=d.task_valid_rows[tile*4]
  assert all(x==e for x in d.task_expert[tile*4:tile*4+4])
  for j in range(128):
   pos=tile*128+j;t=d.token_map[pos];s=d.route_slots[pos]
   if j<valid:assert ids[t][s]==e;seen.append((t,s))
   else:assert t==s==-1
 assert sorted(seen)==[(t,s) for t in range(m) for s in range(2)]
 assert intermediate_bytes(len(d.token_map))==len(d.token_map)*288
for k in (512,2048):
 for rows in (128,256):
  offsets=[scale_offset(r,c,k) for r in range(rows) for c in range(k//16)]
  assert sorted(offsets)==list(range(rows*k//16))
try:publish([[1,1]])
except ValueError:pass
else:raise AssertionError('duplicate route accepted')
print('domain coverage/bounds/scale-bijection passed')
