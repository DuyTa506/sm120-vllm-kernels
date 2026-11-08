"""Capture actual vLLM FP8 operator operands; no candidate kernel is installed."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-gpu', action='store_true')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--tokens', type=int, default=1024)
    ap.add_argument('--layers', default='0,9,29')
    args = ap.parse_args()
    if not args.run_gpu:
        print('CPU preflight only'); return
    assert not args.out.exists()
    os.environ.setdefault('VLLM_ENABLE_V1_MULTIPROCESSING', '0')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    import torch
    from vllm import LLM, SamplingParams
    model_id = 'unsloth/Qwen3.6-35B-A3B-NVFP4-Fast'
    snapshots = sorted(Path('/hf/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots').glob('*'))
    assert len(snapshots) == 1, 'Explicit snapshot selection needed'
    snapshot = snapshots[0]
    record = {'status':'running', 'purpose':'capture-only exact runtime operands; pristine image eager/noMTP, not composed scalar+next serving', 'model':model_id,
              'snapshot':snapshot.name, 'checkpoint_metadata_sha256':{}, 'captures':[]}
    for name in ('config.json','model.safetensors.index.json'):
        record['checkpoint_metadata_sha256'][name] = hashlib.sha256((snapshot/name).read_bytes()).hexdigest()
        (args.out.parent/('checkpoint_'+name)).write_bytes((snapshot/name).read_bytes())
    args.out.write_text(json.dumps(record,indent=2)+'\n')
    llm = LLM(model=str(snapshot), enforce_eager=True, optimization_level=0,
              max_model_len=args.tokens+64, max_num_batched_tokens=args.tokens+64,
              gpu_memory_utilization=0.65, enable_prefix_caching=False,
              language_model_only=True, max_num_seqs=4, seed=1729)
    ids = llm.get_tokenizer()(Path('/harness/prompt.txt').read_text())['input_ids'][:args.tokens]
    assert len(ids) == args.tokens, 'Real-text prompt too short'
    wanted = set(map(int,args.layers.split(',')))
    out_dir = str(args.out.parent)
    min_rows = args.tokens
    def install(model):
        import re
        import inspect
        import hashlib
        import json
        from pathlib import Path
        import torch
        from vllm import _custom_ops as ops
        by_pointer = {}
        runtime_sources = set()
        for name,module in model.named_modules():
            found = re.search(r'layers\.(\d+)\.',name)
            if not found or int(found.group(1)) not in wanted:
                continue
            if 'linear_attn' not in name or not name.endswith('in_proj_qkvz'):
                continue
            weight = getattr(module,'weight',None)
            if weight is not None:
                for obj in (type(module),type(getattr(module,'quant_method',None))):
                    source_file = inspect.getsourcefile(obj)
                    if source_file: runtime_sources.add(source_file)
                by_pointer[weight.data_ptr()] = {'module':name, 'class':type(module).__module__+'.'+type(module).__name__, 'layer':int(found.group(1)), 'runtime_weight_pointer':weight.data_ptr(), 'runtime_weight_shape':list(weight.shape), 'runtime_weight_stride':list(weight.stride()), 'quant_method_class':type(getattr(module,'quant_method',None)).__module__+'.'+type(getattr(module,'quant_method',None)).__name__}
        assert {x['layer'] for x in by_pointer.values()} == wanted, ('GDN modules not found',by_pointer)
        original = ops.cutlass_scaled_mm
        signature = inspect.signature(original)
        captures = []
        done = set()
        def meta(t):
            return {'shape':list(t.shape),'stride':list(t.stride()),'dtype':str(t.dtype),
                    'sha256_logical_bytes':hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()}
        def wrapped(*pos,**kw):
            bound = signature.bind(*pos,**kw); bound.apply_defaults()
            values = bound.arguments
            a,b = values['a'],values['b']
            result = original(*pos,**kw)
            desc = by_pointer.get(b.data_ptr())
            if desc is not None and desc['layer'] not in done and a.shape[0] >= min_rows:
                sa,sb = values['scale_a'],values['scale_b']
                assert values.get('bias') is None, 'Bias unsupported by fused candidate'
                assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn
                assert tuple(b.shape) == (2048,12288) and a.is_contiguous()
                assert b.stride(0) == 1 and b.stride(1) == b.shape[0], 'Unsupported runtime weight layout'
                assert result.dtype == torch.bfloat16 and sa.numel() == a.shape[0] and sb.numel() == b.shape[1]
                tensors = {key:t.detach().cpu().clone(memory_format=torch.preserve_format)
                           for key,t in {'a':a,'b':b,'sa':sa,'sb':sb,'output':result}.items()}
                metadata = dict(desc, operands={key:meta(t) for key,t in tensors.items()}, bias=None)
                filename = 'gdn_qkvz_layer'+str(desc['layer'])+'.pt'
                torch.save({'tensors':tensors,'metadata':metadata},Path(out_dir)/filename)
                metadata['file']=filename
                metadata['file_sha256']=hashlib.sha256((Path(out_dir)/filename).read_bytes()).hexdigest()
                captures.append(metadata); done.add(desc['layer'])
            return result
        ops.cutlass_scaled_mm = wrapped
        model._gdn_capture = captures
        source = Path(inspect.getsourcefile(original))
        runtime_sources.add(str(source))
        runtime_sources.add(str(source.parent/'model_executor/kernels/linear/scaled_mm/cutlass.py'))
        source_records = []
        for source_name in sorted(runtime_sources):
            source_path = Path(source_name)
            target = Path(out_dir)/'runtime_sources'/source_path.relative_to('/usr/local/lib/python3.12/dist-packages')
            target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(source_path.read_bytes())
            source_records.append({'path':source_name,'sha256':hashlib.sha256(source_path.read_bytes()).hexdigest()})
        return {'runtime_sources':source_records, ** {'matched_modules':list(by_pointer.values()),'vllm_custom_ops_file':str(source),
                'vllm_custom_ops_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}}
    try:
        record['installed'] = llm.apply_model(install)[0]
        llm.generate([{'prompt_token_ids':ids}], SamplingParams(max_tokens=1,temperature=0), use_tqdm=False)
        record['captures'] = llm.apply_model(lambda m:m._gdn_capture)[0]
        assert len(record['captures']) == len(wanted), 'Not all requested actual operands captured'
        record['prompt_tokens_sha256'] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        record['prompt_text_sha256'] = hashlib.sha256(Path('/harness/prompt.txt').read_bytes()).hexdigest()
        record['status']='passed'
    except BaseException:
        import traceback
        record['status']='failed';record['error']=traceback.format_exc();raise
    finally:
        args.out.write_text(json.dumps(record,indent=2)+'\n')

if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import sys
        import traceback
        if '--out' in sys.argv:
            output = Path(sys.argv[sys.argv.index('--out')+1])
            if output.exists():
                data=json.loads(output.read_text());data['status']='failed';data['error']=traceback.format_exc()
                output.write_text(json.dumps(data,indent=2)+'\n')
        raise
