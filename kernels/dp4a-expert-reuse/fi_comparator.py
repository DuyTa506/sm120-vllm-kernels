"""Explicit calibrated FI comparator; bypass wrapper's omitted reciprocal flag.
No source edits and no unit-scale substitution. Numerical diagnostic only.
"""
def run(dispatch,expert,raw,prep,x,ids,weights,out):
 return dispatch.launch_sm120_moe(
  a=x,topk_ids=ids,topk_weights=weights,
  w1_weight=prep['w13'],w1_weight_sf=expert.w1_sf_mma,
  w1_alpha=raw['wgs13'].reciprocal().contiguous(),
  input_global_scale=prep['a13'].expand(256).contiguous(),
  fc2_input_scale=prep['a2'].expand(256).contiguous(),
  w2_weight=prep['w2'],w2_weight_sf=expert.w2_sf_mma,w2_alpha=prep['g2'],
  num_experts=256,top_k=8,num_local_experts=256,scatter_output=out,
  input_scales_are_reciprocal=True,activation='silu',activation_precision='fp4',
  quant_mode='nvfp4',source_format='modelopt')
