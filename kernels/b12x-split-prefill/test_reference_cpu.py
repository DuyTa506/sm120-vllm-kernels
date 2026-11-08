import torch
from numerical_reference import decode,unswizzle,metrics
from domain import scale_offset
q=torch.tensor([[0x10,0x32,0x54,0x76,0x98,0xba,0xdc,0xfe]*8],dtype=torch.uint8)
s=torch.ones((1,8),dtype=torch.float8_e4m3fn)
d=decode(q,s)
expected=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],dtype=torch.float64).repeat(8)[None,:]
assert torch.equal(d,expected)
for rows in (1,65,128,129):
 k=512;linear=torch.arange(rows*32).reshape(rows,32).remainder(126).to(torch.uint8)
 flat=torch.zeros(((rows+127)//128)*128*32,dtype=torch.uint8)
 for r in range(rows):
  for b in range(32):flat[scale_offset(r,b,k)]=linear[r,b]
 assert torch.equal(unswizzle(flat,rows,k).view(torch.uint8),linear)
assert metrics(torch.zeros(2),torch.zeros(2))['rel_l2']==0
assert not torch.cuda.is_initialized()
print('reference nibble decode, independent swizzle inverse/tails, zero norm passed on CPU')
