"""Prepared explicit FI direct launch; no process-global dispatch mutation.

CPU-prepared candidate. Caller owns calibrated operands and private workspace.
Constructor is setup/JIT work and must run before graph capture. Runtime input
shape/dtype/addresses are fixed by the prepared bindings. Not a serving flag yet.
"""
class DirectNative:
 def __init__(self, dispatch, plan):
  if not plan.direct_multiplier or plan.forced != 'direct_micro':
   raise ValueError('requires calibrated preplanned multiplier direct Plan')
  kw=plan.kw; x=kw['a'];ids=kw['topk_ids'];weights=kw['topk_weights'];ws=plan.workspace;wv=plan.views
  m=x.shape[0]
  if m not in (1,3,4,8) or x.shape[1]!=2048 or ids.shape!=(m,8):
   raise ValueError('unsupported direct candidate shape')
  compiled,grid,accepted=dispatch._get_direct_micro_kernel(
   256,m,2048,512,8,topk_ids_dtype=ids.dtype,fast_math=True,
   share_input_across_experts=False,share_expert_scales=False,activation='silu',device=x.device)
  if not accepted:raise RuntimeError('native direct launch configuration rejected')
  self.compiled=compiled;self.launch=dispatch.MoEDirectMicroKernel.launch
  self.kw=dict(x=x,w1_fp4=wv.w1_storage,w1_blockscale=wv.w1_scale_storage,
   w1_alphas=wv.w1_alpha,a1_gscale=plan.a1,a2_gscale=plan.a2,
   inter_fp32=ws.dm_intermediate,w2_fp4=wv.w2_storage,w2_blockscale=wv.w2_scale_storage,
   w2_alphas=wv.w2_alpha,topk_ids=ids.view(-1),topk_weights=weights.view(-1),
   out=kw['scatter_output'],barrier_count=ws.dm_barrier_count,
   barrier_epoch=ws.dm_barrier_epoch,m=m,grid_x=grid)
 def __call__(self,out=None):
  # Per-call dictionary, never mutate shared prepared bindings.
  target=self.kw['out'] if out is None else out
  self.launch(self.compiled,**dict(self.kw,out=target))
  return target
