"""Isolated stage diagnosis, no timing or promotion; explicit RU/RN experiment."""
import argparse,json,os,sys,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--tokens',type=int,default=1024);p.add_argument('--layer',type=int,default=19);p.add_argument('--layout',default='indexed');p.add_argument('--rounding',choices=['ru','rn'],required=True);a=p.parse_args()
assert a.layout=='indexed' and a.tokens==1024 and a.capture.stat().st_size>1000000
for k,v in dict(VLLM_B12X_FIX_SCALES='1',VLLM_B12X_SHARED_WORKSPACE='1',VLLM_B12X_SCALAR_INPUT_SCALE='1',VLLM_B12X_HYBRID_MIN_TOKENS='1024',VLLM_B12X_ACT_SCALES='static',VLLM_HUMMING_MOE_GEMM_TYPE='indexed').items():os.environ[k]=v
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'moe-nvfp4'))
import torch
from rounding_mode import install
rounding=install(a.rounding)
from moe_common import load_layer_raw,prep_cutlass,route,reference,dequant,quant_input
from verify_shared_workspace import apply,errors,load_backend
from verify_scalar_dispatch import prepare_exact
from test_shared_workspace import make_expert
from prepare import prepare
from vllm.v1.worker import workspace
from vllm import _custom_ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from humming.testing.process_input import unpack_group_scales
from humming import ops as humming_ops
import humming,inspect
package=Path(humming.__file__).parent
files=[package/'include/humming/kernel/process_input/quantization.cuh',Path(inspect.getsourcefile(_custom_ops)),Path('/usr/local/lib/python3.12/dist-packages/flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_w4a16_fp4_helpers.py')]
files+=list(Path('/usr/local/lib/python3.12/dist-packages/vllm').glob('_C*.so'))
base=Path(__file__).resolve().parent.parent/'moe-nvfp4'
files += [base/n for n in ('moe_common.py','verify_shared_workspace.py','verify_scalar_dispatch.py','test_shared_workspace.py','scalar/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py')]
files += list(package.rglob('*.py')) + list(package.rglob('*.cuh'))
files += [Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py'),Path('/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/humming/moe.py')]
files = sorted(set(files))

source_sha={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(a.output.parent/'runtime_sources.json').write_text(json.dumps(source_sha,indent=2))
import shutil
archive=a.output.parent/'bound_helpers';archive.mkdir(exist_ok=False)
for path in files:
    if base in path.parents:
        dest=archive/path.relative_to(base);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
workspace.init_workspace_manager(torch.device('cuda:0'))
torch.backends.cuda.matmul.allow_tf32=False
raw=load_layer_raw(a.layer);prep=prep_cutlass(raw)
scalar=prepare_exact(load_backend('scalar'),raw,prep,make_expert)
layer,expert,make_call=prepare(raw,'indexed',a.layer)
x=torch.load(a.capture,map_location='cpu',weights_only=True)['hidden'][a.layer][:a.tokens].cuda().contiguous();weights,ids=route(x,raw['gate'])
result=dict(rounding=rounding,scope='numerical only; three backends plus independent FP32 dequantized reference',layer=a.layer,tokens=a.tokens,capture_sha256=hashlib.sha256(a.capture.read_bytes()).hexdigest())
def clone(v):return v.clone() if isinstance(v,torch.Tensor) else v
stages={};original_process=expert.process_input;original_forward=expert.humming_forward
def process(name,*args,**kwargs):
    assert kwargs.get('scatter_idx') is None, 'diagnostic requires original flattened routing row order'
    stages[name+'_before_quant']=kwargs['inputs'].clone()
    ans=original_process(name,*args,**kwargs)
    stages[name+'_quant']=tuple(clone(v) for v in ans)
    return ans
def forward(name,*args,**kwargs):
    ans=original_forward(name,*args,**kwargs)
    stages[name+'_gemm_output']=kwargs['outputs'].clone()
    return ans
expert.process_input=process;expert.humming_forward=forward
outputs={name:torch.empty_like(x) for name in ['scalar','cutlass','humming']}
apply(scalar,prep,x,ids,weights,outputs['scalar'])
scalar._apply_cutlass(outputs['cutlass'],x,prep['w13'],prep['w2'],weights,ids,MoEActivation.SILU)
make_call(x,ids,weights,outputs['humming'])()
# FP32 reference uses original checkpoint packed bytes, original E4M3 local
# scales and global multipliers, unquantized real input, independent LUT decode.
yref=reference(x,raw,ids,weights)
result['full_output_vs_fp32']={name:errors(y,yref) for name,y in outputs.items()}
result['full_pairwise']={f'{left}_vs_{right}':errors(outputs[left],outputs[right]) for left,right in [('humming','scalar'),('humming','cutlass'),('scalar','cutlass')]}
LUT=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6],device='cuda',dtype=torch.float32)
def payload_decode(packed):
    b=packed.contiguous().view(torch.uint8)
    return torch.stack((LUT[(b&15).long()],LUT[(b>>4).long()]),-1).flatten(-2)
def decode_humming(name):
    q,s,global_scale=stages[name+'_quant']; values=payload_decode(q)
    scale=unpack_group_scales(s,values.shape[0],values.shape[1]//16,'float8e4m3',False)
    return (values.view(values.shape[0],-1,16)*scale.float().unsqueeze(-1)*global_scale.float().reshape(1,1,1)).flatten(1),q,scale
xq,q1,s1=decode_humming('w13');zq,q2,s2=decode_humming('w2')
# Layout-normal FI/vLLM quantizer; same calibration, no swizzled-byte comparison.
fi_q,fi_s=_custom_ops.scaled_fp4_quant(x,prep['a1_gscale'][:1].contiguous(),is_sf_swizzled_layout=False)
fi_s=fi_s.view(torch.float8_e4m3fn)[:a.tokens,:x.shape[1]//16]
def bytes_compare(left,right):
    left=left.contiguous().view(torch.uint8);right=right.contiguous().view(torch.uint8)
    assert left.shape==right.shape,(left.shape,right.shape)
    return dict(exact=bool(torch.equal(left,right)),mismatch_fraction=float((left!=right).float().mean()))
result['fc1_quant_payload_vs_vllm']=bytes_compare(q1.view(torch.uint8),fi_q.view(torch.uint8))
result['fc1_quant_scale_vs_vllm']=bytes_compare(s1,fi_s)
result['fc1_dequant_input_vs_real']=errors(xq,x)
# Isolate weight decode and GEMM math using the EXACT quantized Humming operands.
W13=dequant(raw['w13'],raw['s13'],raw['wgs13']);W2=dequant(raw['w2'],raw['s2'],raw['wgs2'])
flat_ids=ids.flatten().long();tokens=torch.arange(a.tokens,device='cuda').repeat_interleave(8)
fc1_ref=torch.empty((a.tokens*8,1024),device='cuda');fc2_ref=torch.empty((a.tokens*8,2048),device='cuda')
for e in range(256):
    rows=(flat_ids==e).nonzero().flatten()
    if rows.numel():
        fc1_ref[rows]=xq[tokens[rows]]@W13[e].T
        fc2_ref[rows]=zq[rows]@W2[e].T
result['fc1_gemm_given_exact_humming_quant_vs_fp32']=errors(stages['w13_gemm_output'],fc1_ref)
result['fc2_gemm_given_exact_humming_quant_vs_fp32']=errors(stages['w2_gemm_output'].reshape_as(fc2_ref),fc2_ref)
gateup=stages['w2_before_quant'].float()
activation=torch.nn.functional.silu(gateup[:,:512])*gateup[:,512:]
result['fc2_dequant_input_vs_activation']=errors(zq,activation)
from quant_oracle import quantize
for name,source,payload,scales in [('fc1',x,q1,s1),('fc2',activation,q2,s2)]:
    dequant_scale=getattr(layer,('w13' if name=='fc1' else 'w2')+'_input_scale_2')
    for rounding_name in ('ru','rn'):
        oracle_payload,oracle_scale=quantize(source,dequant_scale,rounding_name)
        result[name+'_payload_vs_'+rounding_name+'_oracle']=bytes_compare(payload.view(torch.uint8),oracle_payload)
        result[name+'_scale_vs_'+rounding_name+'_oracle']=bytes_compare(scales,oracle_scale)
# vLLM standalone quantizer requires BF16, so this comparison explicitly adds
# activation rounding; the float32 oracle above isolates that difference.
fi_q2,fi_s2=_custom_ops.scaled_fp4_quant(activation.to(torch.bfloat16),prep['a2_gscale'][:1].contiguous(),is_sf_swizzled_layout=False)
result['fc2_payload_vs_vllm_after_bf16_cast']=bytes_compare(q2.view(torch.uint8),fi_q2.view(torch.uint8))
result['fc2_scale_vs_vllm_after_bf16_cast']=bytes_compare(s2,fi_s2.view(torch.float8_e4m3fn)[:activation.shape[0],:32])
# Packed checkpoint interpretation sanity independent of the transformed kernel.
# This decodes raw source packing with Humming's own operator and independent LUT.
small=raw['w13'][:1,:8].contiguous().view(torch.int32)
result['raw_weight_nibble_decode']=errors(humming_ops.dequant_weight(humming_ops.unpack_weight(small,4),2,1,True),payload_decode(small))
result['all_outputs_finite']=all(bool(torch.isfinite(y).all()) for y in outputs.values())
assert result['all_outputs_finite']
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in source_sha.items())
a.output.write_text(json.dumps(result,indent=2,allow_nan=False))
