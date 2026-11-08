"""Post-call references and domain validation; intentionally synchronizing, never timed."""
import torch
from domain import scale_offset

def metrics(a,b):
 a=a.double();b=b.double();delta=a-b;norm=b.norm()
 return dict(finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),rel_l2=float(delta.norm()/norm.clamp_min(1e-30)),max_abs=float(delta.abs().max()),mismatch_count=int((a!=b).sum()),elements=a.numel(),reference_absmax=float(b.abs().max()))

def validate_domain(w,ids,weights):
 assert int(w.errors)==0,'publisher invalid-ID/duplicate flag'
 ids=ids.cpu();weights=weights.cpu();counts=w.counts.cpu();bases=w.tile_bases.cpu();tm=w.token_map.cpu();slots=w.route_slots.cpu();te=w.task_expert.cpu();vr=w.task_rows.cpu()
 expected=torch.bincount(ids[ids>=0].long(),minlength=256)
 assert torch.equal(counts,expected) and torch.equal(w.write_rows.cpu(),counts)
 assert torch.equal(bases[1:]-bases[:-1],(counts+127)//128)
 assert int(bases[-1])<=w.capacity_tiles and int(counts.sum())==int((ids>=0).sum())
 pa=w.packed_a.cpu().reshape(-1,1024);sa=w.scale_a.cpu();q=w.q.cpu();sf=w.sf_bytes.cpu();tw=w.token_weights.cpu()
 seen=[]
 for e in range(256):
  for tile in range(int(bases[e]),int(bases[e+1])):
   n=min(128,int(counts[e])-(tile-int(bases[e]))*128)
   assert torch.equal(te[tile*4:tile*4+4],torch.full((4,),e))
   assert torch.equal(vr[tile*4:tile*4+4],torch.full((4,),n))
   rows=torch.arange(tile*128,tile*128+n);tokens=tm[rows].long();ss=slots[rows].long()
   assert bool((ids[tokens,ss]==e).all());assert torch.equal(tw[rows],weights[tokens,ss]);assert torch.equal(pa[rows],q[tokens])
   blocks=torch.arange(128)[None,:];rp=rows[:,None];tp=tokens[:,None]
   dst=(rp//128)*16384+(blocks//4)*512+(rp%32)*16+((rp%128)//32)*4+blocks%4
   src=(tp//128)*16384+(blocks//4)*512+(tp%32)*16+((tp%128)//32)*4+blocks%4
   assert torch.equal(sa[dst],sf[src])
   seen.extend(zip(tokens.tolist(),ss.tolist()))
   tail=slice(tile*128+n,(tile+1)*128)
   assert bool((tm[tail]==-1).all()) and bool((slots[tail]==-1).all()) and not bool(pa[tail].any())
   if n<128:
    tailrows=torch.arange(tile*128+n,(tile+1)*128)[:,None]
    tailoff=(tailrows//128)*16384+(blocks//4)*512+(tailrows%32)*16+((tailrows%128)//32)*4+blocks%4
    assert not bool(sa[tailoff].any())
 expected_pairs=[(t,s) for t in range(w.m) for s in range(8) if ids[t,s]>=0]
 assert sorted(seen)==expected_pairs,'missing/duplicate routed contribution'
 return dict(passed=True,routed_rows=len(seen),active_tiles=int(bases[-1]),capacity_tiles=w.capacity_tiles,pack_and_sfa_byte_exact=True,row_order='canonical by token/route slot; physical atomic placement may differ')

def decode(q,s):
 lut=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],device=q.device,dtype=torch.float64)
 nib=torch.stack((q&15,q>>4),-1).long().flatten(-2)
 return (lut[nib].reshape(q.shape[0],-1,16)*s.double()[...,None]).flatten(-2)

def unswizzle(sf,rows,k):
 row=torch.arange(rows,device=sf.device)[:,None];block=torch.arange(k//16,device=sf.device)[None,:]
 idx=(row//128)*(k//64)*512+(block//4)*512+(row%32)*16+((row%128)//32)*4+block%4
 return sf.flatten()[idx].contiguous().view(torch.float8_e4m3fn)

def reference_check(w,prep,raw,x,ids,weights):
 from vllm import _custom_ops as ops
 # Independent input quantization comparison, including changed input after replay.
 q,sf=ops.scaled_fp4_quant(x,prep['a1_gscale'][:1])
 assert torch.equal(q,w.q) and torch.equal(sf.view(torch.uint8).flatten(),w.sf_bytes),'input quantizer differs/stale'
 rowscap=w.capacity_tiles*128;midbytes=w.intermediate.view(torch.uint8)
 mq=midbytes[:rowscap*256].reshape(rowscap,256)
 ms=midbytes[rowscap*256:].reshape(4,rowscap,8).permute(1,0,2).contiguous().reshape(rowscap,32).view(torch.float8_e4m3fn)
 input_units=decode(w.q,unswizzle(w.sf_bytes,w.m,2048))
 counts=w.counts.cpu().tolist();bases=w.tile_bases.cpu().tolist();all_mid=[];all_ref_mid=[];per_expert=[];phase1_bits=0;phase1_scales=0;canonical_mismatch=0;signedzero_only=0;zero_scale_mismatch=0;output_ref=torch.zeros_like(w.output,dtype=torch.float64)
 # Full observed routed-domain reference, not a few cherry-picked rows. FP64
 # accumulation isolates MMA/epilogue errors; BF16 activation rounding explicit.
 for e,count in enumerate(counts):
  if not count:continue
  rr=slice(bases[e]*128,bases[e]*128+count);tokens=w.token_map[rr].long()
  w13q=torch.cat((raw['w13'][e,512:],raw['w13'][e,:512]),0)
  w13sf=torch.cat((raw['s13'][e,512:],raw['s13'][e,:512]),0)
  h=(input_units[tokens]@decode(w13q,w13sf).T)*prep['g1'][e].double()
  activation=(torch.nn.functional.silu(h[:,512:])*h[:,:512]).to(torch.bfloat16).contiguous()
  rq,rsf=ops.scaled_fp4_quant(activation,prep['a2_gscale'][e:e+1])
  rs=unswizzle(rsf.view(torch.uint8),count,512)
  actual_mid=decode(mq[rr],ms[rr]);expected_mid=decode(rq,rs)
  all_mid.append(actual_mid);all_ref_mid.append(expected_mid);per_expert.append(dict(expert=e,rows=count,phase1=metrics(actual_mid,expected_mid)))
  phase1_bits+=int((mq[rr]!=rq).sum());phase1_scales+=int((ms[rr].view(torch.uint8)!=rs.view(torch.uint8)).sum())
  def canonical(q):
   lo=q&15;hi=q>>4
   return torch.where(lo==8,0,lo)|((torch.where(hi==8,0,hi))<<4)
  raw_mismatch=mq[rr]!=rq;canonical_diff=canonical(mq[rr])!=canonical(rq)
  same_sf=ms[rr].view(torch.uint8)==rs.view(torch.uint8)
  positive_sf=(ms[rr].view(torch.uint8)!=0)&same_sf
  positive_bytes=positive_sf.repeat_interleave(8,dim=1)
  zero_bytes=((ms[rr].view(torch.uint8)==0)&same_sf).repeat_interleave(8,dim=1)
  canonical_mismatch+=int(canonical_diff.sum())
  signedzero_only+=int((raw_mismatch&~canonical_diff&positive_bytes).sum())
  zero_scale_mismatch+=int((raw_mismatch&zero_bytes).sum())
  # FC2 oracle consumes ACTUAL produced intermediate so phase1 differences do
  # not get misattributed to FC2. Fixed expert-order FP64 reduction.
  scale=(prep['g2'][e]*w.token_weights[rr]).double()
  y=(actual_mid@decode(raw['w2'][e],raw['s2'][e]).T)*scale[:,None]
  output_ref.index_add_(0,tokens,y)
 if all_mid:p1=metrics(torch.cat(all_mid),torch.cat(all_ref_mid))
 else:p1=dict(finite=True,rel_l2=0.,max_abs=0.,mismatch_count=0,elements=0,reference_absmax=0.)
 p2=metrics(w.output,output_ref)
 return dict(passed=p1['finite'] and p2['finite'] and p1['rel_l2']<=.01 and p2['rel_l2']<=.02,input_quant_bytes_exact=True,phase1_vs_fp64_bf16_requant=p1,phase1_payload_byte_mismatch=phase1_bits,phase1_scale_byte_mismatch=phase1_scales,phase1_canonical_payload_byte_mismatch=canonical_mismatch,phase1_positive_scale_signedzero_only_byte_mismatch=signedzero_only,phase1_zero_scale_byte_mismatch=zero_scale_mismatch,per_expert_phase1=per_expert,phase2_vs_actual_mid_fp64=p2,scope='Full valid routed domain, FP64 accumulation; exploratory thresholds, not production acceptance')
