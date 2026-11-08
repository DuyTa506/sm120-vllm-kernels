import flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch as dispatch
class MMA:
 def __init__(self,expert,prep,x,ids,w,out,backend):
  self.backend=backend
  self.ws=dispatch.allocate_sm120_static_workspace(state_E=256,weight_E=256,max_rows=x.shape[0]*8,k=2048,n=512,num_topk=8,device=x.device,activation_precision='fp4',quant_mode='nvfp4')
  views=dispatch._get_weight_views(w1_fp4=prep['w13'],w1_blockscale=expert.w1_sf_mma,w2_fp4=prep['w2'],w2_blockscale=expert.w2_sf_mma,w1_alphas=prep['g1'],w2_alphas=prep['g2'],n=512,k=2048,activation_precision='fp4',quant_mode='nvfp4')
  self.kw=dict(a=x,topk_ids=ids,topk_weights=w,w1_weight=prep['w13'],w1_weight_sf=expert.w1_sf_mma,w1_alpha=prep['g1'],input_global_scale=prep['a13'].expand(256).contiguous(),fc2_input_scale=prep['a2'].expand(256).contiguous(),w2_weight=prep['w2'],w2_weight_sf=expert.w2_sf_mma,w2_alpha=prep['g2'],num_experts=256,top_k=8,num_local_experts=256,scatter_output=out,input_scales_are_reciprocal=False,activation='silu',activation_precision='fp4',quant_mode='nvfp4',source_format='modelopt',_workspace=self.ws,_weight_views=views)
 def __call__(self):
  old=dispatch._FORCED_BACKEND;dispatch._FORCED_BACKEND=self.backend
  try:return dispatch.launch_sm120_moe(**self.kw)
  finally:dispatch._FORCED_BACKEND=old
