"""Audit all 40 checkpoint MoE FC1 scalar-selection guards, on CPU only.

Run in the pinned container with CUDA_VISIBLE_DEVICES= and PYTHONDONTWRITEBYTECODE=1.
Reads only scalar input tensors and safetensors headers, not packed model weights.
The executed expressions/guard are extracted from the installed preparation chain
and the scalar overlay; GPU block-scale layout conversion is outside this audit.
"""
import argparse
import ast
from contextlib import ExitStack
from enum import Enum
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
OVERLAY = HERE / 'scalar/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py'
SNAPSHOTS = Path('/root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots')


def sha(data): return hashlib.sha256(data).hexdigest()


def node(path, name, class_name=None):
    tree=ast.parse(path.read_text())
    body=tree.body if class_name is None else next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name==class_name).body
    return next(x for x in body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==name)


def execute(nodes, namespace, filename):
    tree=ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[]))
    exec(compile(tree,str(filename),'exec'),namespace)


def expression(node, namespace, filename):
    return eval(compile(ast.Expression(body=node),str(filename),'eval'),namespace)


def describe(tensor):
    tensor=tensor.detach().contiguous().cpu()
    return {'dtype':str(tensor.dtype),'shape':list(tensor.shape),'sha256':sha(tensor.numpy().tobytes()),
            'finite':bool(torch.isfinite(tensor).all()),'positive':bool((tensor>0).all()),
            'uniform':bool(torch.equal(tensor.reshape(-1),tensor.reshape(-1)[:1].expand(tensor.numel()))),
            'unique_values':int(torch.unique(tensor).numel()),'min':float(tensor.min()),'max':float(tensor.max())}


def audit(snapshot):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU audit requires CUDA_VISIBLE_DEVICES=')
    if torch.cuda.is_initialized():raise RuntimeError('CUDA was already initialized')
    torch.set_num_threads(1)
    root=Path(importlib.util.find_spec('vllm').origin).parent
    paths={
        'compressed_preparation':root/'model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_nvfp4.py',
        'format_preparation':root/'model_executor/layers/quantization/utils/flashinfer_fp4_moe.py',
        'backend_quant_config':root/'model_executor/layers/fused_moe/oracle/nvfp4.py',
        'activation_reduction':root/'model_executor/layers/quantization/utils/quant_utils.py',
        'scalar_overlay':OVERLAY,'audit_script':Path(__file__),
    }
    config=json.loads((snapshot/'config.json').read_text()); text_config=config['text_config']
    assert text_config['num_hidden_layers']==40 and text_config['num_experts']==256
    index_path=snapshot/'model.safetensors.index.json'; weight_map=json.loads(index_path.read_text())['weight_map']
    matcher=re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.input_global_scale$')
    found={(int(m[1]),int(m[2]),m[3]) for key in weight_map if (m:=matcher.fullmatch(key))}
    expected={(layer,expert,projection) for layer in range(40) for expert in range(256) for projection in ('gate','up','down')}
    assert found==expected, 'Incomplete/unexpected target checkpoint input-scale set'
    namespace={'torch':torch,'Enum':Enum,'os':SimpleNamespace(environ={
        'VLLM_B12X_ACT_SCALES':'static','VLLM_B12X_HYBRID_MIN_TOKENS':'1024'}),
        '_B12X_FIX_SCALES':True,'_B12X_SCALAR_INPUT_SCALE':True}
    oracle=ast.parse(paths['backend_quant_config'].read_text())
    backend_list=next(x for x in oracle.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FLASHINFER_NVFP4_MOE_BACKENDS' for t in x.targets))
    execute([node(paths['backend_quant_config'],'NvFp4MoeBackend'),backend_list,
             node(paths['backend_quant_config'],'is_global_sf_supported_for_nvfp4_backend')],namespace,paths['backend_quant_config'])
    namespace['backend']=namespace['NvFp4MoeBackend'].FLASHINFER_B12X
    assert namespace['is_global_sf_supported_for_nvfp4_backend'](namespace['backend'])
    execute([node(paths['activation_reduction'],'amax_for_moe_activation_quant')],namespace,paths['activation_reduction'])
    compressed=node(paths['compressed_preparation'],'process_weights_after_loading','CompressedTensorsW4A4Nvfp4MoEMethod')
    conversion=next(x for x in ast.walk(compressed) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='convert_to_nvfp4_moe_kernel_format')
    load_expr={k.arg:k.value for k in conversion.keywords if k.arg in ('a13_scale','a2_scale')}
    prepare=node(paths['format_preparation'],'prepare_nvfp4_moe_layer_for_fi_or_cutlass')
    reduction=next(x for x in prepare.body if isinstance(x,ast.If) and isinstance(x.test,ast.Call) and isinstance(x.test.func,ast.Name) and x.test.func.id=='is_global_sf_supported_for_nvfp4_backend')
    quant=node(paths['backend_quant_config'],'make_nvfp4_moe_quant_config')
    quant_return=quant.body[-1]
    assert isinstance(quant_return,ast.Return) and isinstance(quant_return.value,ast.Call)
    quant_expr={k.arg:k.value for k in quant_return.value.keywords if k.arg in ('a1_gscale','a2_gscale')}
    process=node(OVERLAY,'_process_weights_fixed_scales','FlashInferB12xExperts')
    activation_branch=next(x for x in process.body if isinstance(x,ast.If) and 'VLLM_B12X_ACT_SCALES' in ast.unparse(x.test))
    selected=[]
    for x in process.body:
        if isinstance(x,ast.Assign) and any((isinstance(t,ast.Name) and t.id in ('dev','hybrid_min')) or
                (isinstance(t,ast.Attribute) and t.attr in ('_input_global_scale','_hybrid_min_tokens')) for t in x.targets): selected.append(x)
        elif x is activation_branch:selected.append(x)
    assert len(selected)==5, 'Overlay scale preparation changed: review dependency selection'
    execute([node(OVERLAY,'_maybe_use_scalar_input_scale','FlashInferB12xExperts')],namespace,OVERLAY)
    assumption={'tp_size':1,'pp_size':1,'ep_size':1,'ep_rank':0,'use_ep':False,'enable_eplb':False,
                'local_experts_equal_global':True,'output_dtype':'torch.bfloat16',
                'VLLM_B12X_FIX_SCALES':'1','VLLM_B12X_SCALAR_INPUT_SCALE':'1',
                'VLLM_B12X_ACT_SCALES':'static','VLLM_B12X_HYBRID_MIN_TOKENS':'1024'}
    rows=[]; headers={}; tensor_hash=hashlib.sha256()
    with ExitStack() as stack:
        handles={}
        def handle(shard):
            if shard not in handles:
                p=snapshot/shard; handles[shard]=stack.enter_context(safe_open(p,framework='pt',device='cpu'))
                with p.open('rb') as stream:
                    prefix=stream.read(8); length=int.from_bytes(prefix,'little'); header=stream.read(length)
                assert len(header)==length
                headers[shard]={'file_size':p.stat().st_size,'cache_blob_name':p.resolve().name,
                    'header_sha256':sha(prefix+header),'full_shard_rehashed':False}
            return handles[shard]
        def get(name):
            t=handle(weight_map[name]).get_tensor(name)
            assert t.device.type=='cpu' and t.numel()==1
            raw=t.contiguous().numpy().tobytes()
            tensor_hash.update(name.encode()+b'\0'+str(t.dtype).encode()+b'\0'+raw)
            return t.to(torch.float32).reshape(())
        for layer_index in range(40):
            prefix=f'model.language_model.layers.{layer_index}.mlp'
            raw={projection:torch.stack([get(f'{prefix}.experts.{expert}.{projection}_proj.input_global_scale') for expert in range(256)])
                 for projection in ('gate','up','down')}
            expected_shapes={'gate':[512,1024],'up':[512,1024],'down':[2048,256]}
            for expert in range(256):
                for projection,shape in expected_shapes.items():
                    name=f'{prefix}.experts.{expert}.{projection}_proj.weight_packed'
                    assert handle(weight_map[name]).get_slice(name).get_shape()==shape, name
            layer=SimpleNamespace(w13_input_global_scale=torch.stack([raw['gate'],raw['up']],dim=1),
                w2_input_global_scale=raw['down'],w13_weight=torch.empty(0),
                moe_config=SimpleNamespace(moe_parallel_config=SimpleNamespace(enable_eplb=False)))
            namespace.update(layer=layer,w13=SimpleNamespace(shape=(256,1024,1024)))
            for key,expr in load_expr.items():namespace[key]=expression(expr,namespace,paths['compressed_preparation'])
            loaded=describe(namespace['a13_scale'])
            execute([reduction],namespace,paths['format_preparation'])
            processed=describe(namespace['a13_scale'])
            obj=SimpleNamespace(global_num_experts=text_config['num_experts'],num_local_experts=256,
                ep_rank=0,topk=text_config['num_experts_per_tok'],hidden_dim=text_config['hidden_size'],
                intermediate_size_per_partition=text_config['moe_intermediate_size'],out_dtype=torch.bfloat16,
                _activation_str=text_config['hidden_act'],moe_config=SimpleNamespace(
                    moe_parallel_config=SimpleNamespace(ep_size=1,use_ep=False)))
            for key,expr in quant_expr.items():setattr(obj,key,expression(expr,namespace,paths['backend_quant_config']))
            namespace['self']=obj
            execute(selected,namespace,OVERLAY)
            before=obj._input_global_scale; scale=describe(before)
            namespace['_maybe_use_scalar_input_scale'](obj)
            after=obj._input_global_scale
            eligible=after.numel()==1 and before.numel()==256
            exact=eligible and torch.equal(before,after.expand_as(before))
            guards={'global_experts_256':obj.global_num_experts==256,'local_experts_equal_global':obj.num_local_experts==obj.global_num_experts,
                    'ep_rank_zero':obj.ep_rank==0,'ep_size_one':obj.moe_config.moe_parallel_config.ep_size==1,
                    'ep_disabled':not obj.moe_config.moe_parallel_config.use_ep,'topk_8':obj.topk==8,
                    'hidden_dim_2048':obj.hidden_dim==2048,'intermediate_per_partition_512':obj.intermediate_size_per_partition==512,
                    'bf16_output':obj.out_dtype==torch.bfloat16,'silu':obj._activation_str=='silu',
                    'hybrid_threshold_at_least_1024':obj._hybrid_min_tokens>=1024,
                    'input_scale_numel_256':before.numel()==256,'finite':scale['finite'],'positive':scale['positive'],
                    'exact_uniform':scale['uniform'],'fix_and_scalar_flags_enabled':True}
            rows.append({'layer':layer_index,'prefix':prefix,'experts':256,'raw_input_global_scales':{key:describe(value) for key,value in raw.items()},
                'gate_up_raw_equal':bool(torch.equal(raw['gate'],raw['up'])),'loaded_gate_up_reciprocals':loaded,
                'processed_layer_w13_input_scale':processed,'overlay_input_global_scale_before_hook':scale,
                'all_packed_weight_shapes_checked':expected_shapes,'guards_under_recorded_assumptions':guards,
                'eligible_under_recorded_assumptions':eligible,'scalar_numel':after.numel(),
                'scalar_equals_all_vector_values':bool(exact),'scalar_independent_storage':after.data_ptr()!=before.data_ptr(),
                'scalar_value':float(after.reshape(-1)[0]),'scalar_fp32_bits_hex':f'{int(after.reshape(-1)[:1].view(torch.int32)[0]) & 0xffffffff:08x}'})
    assert not torch.cuda.is_initialized()
    return {'status':'CPU_AUDIT_COMPLETE','device':'cpu','cuda_initialized':torch.cuda.is_initialized(),'vllm_imported':'vllm' in sys.modules,
        'torch_version':str(torch.__version__),'python_version':sys.version,
        'model':{'id':'unsloth/Qwen3.6-35B-A3B-NVFP4-Fast','snapshot':str(snapshot),'revision':snapshot.name,
            'config_sha256':sha((snapshot/'config.json').read_bytes()),'weight_index_sha256':sha(index_path.read_bytes()),
            'input_scale_tensor_count':len(expected),'ordered_input_scale_tensors_sha256':tensor_hash.hexdigest(),
            'identity_scope':'snapshot revision, config/index, exact consumed scalar bytes, and shard headers; full packed-weight shards not rehashed',
            'shards':headers},
        'source_sha256':{name:{'path':str(path),'sha256':sha(path.read_bytes())} for name,path in paths.items()},
        'executed_expressions':{'checkpoint_scale_reciprocals':{k:ast.unparse(v) for k,v in load_expr.items()},
            'activation_reduction':ast.unparse(reduction),'quant_config_reciprocals':{k:ast.unparse(v) for k,v in quant_expr.items()},
            'overlay_scale_statements':[ast.unparse(x) for x in selected]},
        'guard_assumptions':assumption,'model_config_guards':{key:text_config[key] for key in
            ('num_hidden_layers','num_experts','num_experts_per_tok','hidden_size','moe_intermediate_size','hidden_act','dtype')},
        'summary':{'layers_audited':len(rows),'eligible_layers':sum(x['eligible_under_recorded_assumptions'] for x in rows),
            'finite_positive_uniform_fc1_layers':sum(all(x['overlay_input_global_scale_before_hook'][k] for k in ('finite','positive','uniform')) for x in rows),
            'all_guard_checks_passed_layers':sum(all(x['guards_under_recorded_assumptions'].values()) for x in rows),
            'raw_gate_scale_uniform_layers':sum(x['raw_input_global_scales']['gate']['uniform'] for x in rows)},
        'per_layer':rows,'limitations':[
            'Eligibility under explicit TP1/no-EP/BF16/static-scale/flag/threshold assumptions; this is not observation of live serving layer state or selected GPU dispatch.',
            'The installed B12x preparation takes a maximum across all gate/up expert activation scales and repeats it; raw checkpoint per-expert uniformity is not required.',
            'Only the actual scale-dependency AST and selection hook execute. GPU weight/block-scale layout conversion, workspace tile choice, runtime device support, kernel correctness and speed are not tested.',
            'Small batches remain governed by the existing CUTLASS hybrid threshold; scalar eligibility does not imply every invocation uses B12x.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--snapshot',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();snapshot=args.snapshot
    if snapshot is None:
        choices=list(SNAPSHOTS.iterdir());assert len(choices)==1,'Specify the checkpoint snapshot';snapshot=choices[0]
    result=audit(snapshot);content=json.dumps(result,indent=2,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:stream.write(content)
        print(json.dumps({'status':result['status'],'summary':result['summary'],'output':str(args.output)}))
    else:print(content,end='')

if __name__=='__main__':main()
