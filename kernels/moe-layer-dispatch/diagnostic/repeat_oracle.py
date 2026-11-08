"""Diagnostic only: repeatability decomposition, never a relaxed policy gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
from trace_inputs import describe


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workspace', choices=['shared','private'], required=True)
    p.add_argument('--layer', type=int, default=39)
    p.add_argument('--tokens', type=int, default=1024)
    p.add_argument('--repeats', type=int, default=8)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    base=root/'moe-nvfp4'
    sys.path.insert(0,str(base))
    for key,value in dict(VLLM_B12X_FIX_SCALES='1',VLLM_B12X_SHARED_WORKSPACE=str(int(args.workspace=='shared')),
                          VLLM_B12X_SCALAR_INPUT_SCALE='1',VLLM_B12X_HYBRID_MIN_TOKENS='1024',VLLM_B12X_ACT_SCALES='static').items():
        os.environ[key]=value
    import torch
    from moe_common import load_hidden,load_layer_raw,prep_cutlass,route,H
    from verify_scalar_dispatch import prepare_exact
    from verify_shared_workspace import load_backend,apply,capture,errors
    from test_shared_workspace import make_expert
    from vllm.v1.worker import workspace as ws
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from flashinfer.autotuner import autotune,AutoTuner
    assert args.tokens>=1024 and args.repeats>=3
    args.output.mkdir(parents=True,exist_ok=False)
    ws.init_workspace_manager(torch.device('cuda:0'))
    raw=load_layer_raw(args.layer);prep=prep_cutlass(raw)
    hidden=load_hidden(args.layer)
    x=hidden[:args.tokens].contiguous()
    weights,ids=route(x,raw['gate'])
    expert=prepare_exact(load_backend('scalar'),raw,prep,make_expert)
    out=[torch.empty((args.tokens,H),device='cuda',dtype=torch.bfloat16) for _ in range(2)]
    calls=[lambda:apply(expert,prep,x,ids,weights,out[0]),
           lambda:expert._apply_cutlass(out[1],x,prep['w13'],prep['w2'],weights,ids,MoEActivation.SILU)]
    with autotune(True): calls[1]()
    trace=[]; phase=''
    original=AutoTuner.choose_one
    def choose(self,custom_op,runners,tuning_config,inputs,**kwargs):
        selected,tactic=original(self,custom_op,runners,tuning_config,inputs,**kwargs)
        trace.append(dict(phase=phase,op=custom_op,runner=type(selected).__qualname__,tactic=repr(tactic),
                          inputs=[describe(t) for t in inputs],
                          extras=repr(selected.get_cache_key_extras(inputs))))
        return selected,tactic
    records=[]
    with patch.object(AutoTuner,'choose_one',choose):
        for i,call in enumerate(calls):
            phase=f'backend{i}:capture'
            graph=capture(call)
            saved={}
            for mode in ('eager','graph'):
                values=[]
                for repeat in range(args.repeats):
                    phase=f'backend{i}:{mode}:{repeat}'
                    (call if mode=='eager' else graph.replay)()
                    torch.cuda.synchronize()
                    values.append(out[i].clone())
                saved[mode]=values
                for repeat,value in enumerate(values[1:],1):
                    records.append(dict(backend=i,pair=f'{mode}-{mode}',repeat=repeat,
                                        exact=bool(torch.equal(values[0],value)),**errors(value,values[0])))
            for repeat in range(args.repeats):
                records.append(dict(backend=i,pair='graph-eager',repeat=repeat,
                                    exact=bool(torch.equal(saved['graph'][repeat],saved['eager'][repeat])),
                                    **errors(saved['graph'][repeat],saved['eager'][repeat])))
            del graph,saved
    sources=[Path(__file__),base/'scalar/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py']
    result=dict(workspace=args.workspace,layer=args.layer,tokens=args.tokens,records=records,tactics=trace,
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                interpretation='repeatability diagnostic; does not change policy gates or prove quantization equivalence')
    (args.output/'repeatability.json').write_text(json.dumps(result,indent=2))
    if not all(r['finite'] for r in records): raise SystemExit('nonfinite result')


if __name__=='__main__': main()
