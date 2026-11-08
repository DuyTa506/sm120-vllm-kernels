"""Construct Humming full expert call from original CT checkpoint, GPU-only caller."""
def prepare(raw, layout, layer_number):
    import torch
    from vllm.forward_context import ForwardContext, override_forward_context
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.quantization.utils.humming.moe import convert_to_humming_moe_kernel_format, get_humming_moe_quant_config
    from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import HummingIndexedExperts,HummingGroupedExperts
    from humming.schema.compressed_tensors import CompressedTensorsInputSchema
    from humming import dtypes
    parallel=FusedMoEParallelConfig(tp_size=1,pcp_size=1,dp_size=1,ep_size=1,tp_rank=0,pcp_rank=0,dp_rank=0,ep_rank=0,sp_size=1,use_ep=False,all2all_backend='allgather_reducescatter',enable_eplb=False)
    cfg=FusedMoEConfig(num_experts=256,experts_per_token=8,hidden_dim=2048,intermediate_size=512,num_local_experts=256,num_logical_experts=256,activation=MoEActivation.SILU,device=torch.device('cuda:0'),routing_method=RoutingMethodType.RenormalizeNaive,moe_parallel_config=parallel,in_dtype=torch.bfloat16,moe_backend='humming',max_num_tokens=16384)
    layer=torch.nn.Module();layer.moe_config=cfg;layer.params_dtype=torch.bfloat16
    layer.layer_name=f'model.language_model.layers.{layer_number}.mlp.experts'
    for name, tensor in dict(w13_weight_packed=raw['w13'],w2_weight_packed=raw['w2'],w13_weight_scale=raw['s13'],w2_weight_scale=raw['s2'],w13_weight_global_scale=raw['wgs13'][:,None].expand(-1,2),w2_weight_global_scale=raw['wgs2'],w13_input_global_scale=raw['igs13'][:,None].expand(-1,2),w2_input_global_scale=raw['igs2']).items():
        layer.register_parameter(name,torch.nn.Parameter(tensor.clone().contiguous(),requires_grad=False))
    convert_to_humming_moe_kernel_format(layer,quant_config=dict(quant_method='compressed-tensors',format='nvfp4-pack-quantized',type='float',num_bits=4,strategy='group',group_size=16),input_schema=CompressedTensorsInputSchema(format='nvfp4-pack-quantized',type='float',num_bits=4,dynamic='local',group_size=16),allow_input_schema_fallback=False)
    for name, source in [('w13','igs13'),('w2','igs2')]:
        assert torch.equal(getattr(layer,name+'_input_scale_2'),raw[source].reciprocal().amax().reshape(1))
        schema=layer.input_schemas[name]
        assert schema.a_dtype==dtypes.float4e2m1 and schema.input_scale_group_size==16 and schema.input_quant_mode.value=='static_tensor_dynamic_group'
        c=layer.humming_configs[name]
        assert c.as_dtype==dtypes.float8e4m3, 'activation local scale must stay E4M3'
    cls={'indexed':HummingIndexedExperts,'grouped':HummingGroupedExperts}[layout]
    expert=cls(moe_config=cfg,quant_config=get_humming_moe_quant_config(layer))
    def make_call(x,ids,weights,out):
        shapes=expert._workspace_shapes(x.shape[0],8,MoEActivation.SILU)
        work=[torch.empty(shape,device=x.device,dtype=x.dtype) for shape in shapes[:2]]
        def call():
            # Actual vLLM context: one rank, no DP metadata, supplied routing.
            with override_forward_context(ForwardContext(no_compile_layers={}, attn_metadata={}, slot_mapping={})):
                expert.apply(out,x,layer.w13_weight,layer.w2_weight,weights,ids,MoEActivation.SILU,256,None,None,None,*work,None,False)
            return out
        return call
    return layer,expert,make_call
