"""Preplanned calibrated FI comparator. Immutable scales/views prepared once."""
class Plan:
 def __init__(self,dispatch,expert,prep,x,ids,weights,out,forced,direct_multiplier=False):
  if direct_multiplier and forced!='direct_micro':raise ValueError('multiplier path requires forced direct_micro')
  self.dispatch=dispatch;self.forced=forced
  self.direct_multiplier=direct_multiplier
  self.a1=prep['a13'].expand(256).contiguous();self.a2=prep['a2'].expand(256).contiguous()
  if direct_multiplier:
   self.a1=prep['a1_gscale'].contiguous();self.a2=prep['a2_gscale'].contiguous()
  self.g1=prep['g1'].contiguous();self.g2=prep['g2'].contiguous()
  self.views=dispatch._get_weight_views(w1_fp4=prep['w13'],w1_blockscale=expert.w1_sf_mma,w2_fp4=prep['w2'],w2_blockscale=expert.w2_sf_mma,w1_alphas=self.g1,w2_alphas=self.g2,n=512,k=2048,activation_precision='fp4',quant_mode='nvfp4')
  self.workspace=dispatch.allocate_sm120_static_workspace(state_E=256,weight_E=256,max_rows=x.shape[0]*8,k=2048,n=512,num_topk=8,device=x.device,activation_precision='fp4',quant_mode='nvfp4')
  self.kw=dict(a=x,topk_ids=ids,topk_weights=weights,w1_weight=prep['w13'],w1_weight_sf=expert.w1_sf_mma,w1_alpha=self.g1,input_global_scale=self.a1,fc2_input_scale=self.a2,w2_weight=prep['w2'],w2_weight_sf=expert.w2_sf_mma,w2_alpha=self.g2,num_experts=256,top_k=8,num_local_experts=256,scatter_output=out,input_scales_are_reciprocal=not direct_multiplier,activation='silu',activation_precision='fp4',quant_mode='nvfp4',source_format='modelopt',_workspace=self.workspace,_weight_views=self.views)
 def __call__(self):
  old=self.dispatch._FORCED_BACKEND;self.dispatch._FORCED_BACKEND=self.forced
  try:return self.dispatch.launch_sm120_moe(**self.kw)
  finally:self.dispatch._FORCED_BACKEND=old
