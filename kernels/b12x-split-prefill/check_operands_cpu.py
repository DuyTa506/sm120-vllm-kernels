"""CPU-only checkpoint/phase layout proof. Does not claim GPU execution correctness."""
import argparse, hashlib, json
from pathlib import Path
import torch
from safetensors import safe_open

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--layer',type=int,default=19);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    wm=json.loads((args.snapshot/'model.safetensors.index.json').read_text())['weight_map']; handles={}
    def get(name):
        f=wm[name]
        if f not in handles: handles[f]=safe_open(args.snapshot/f,framework='pt',device='cpu')
        return handles[f].get_tensor(name)
    # Literal scale-address formula used by phase kernels; independently compare
    # against the installed vLLM swizzle reshape/permute (omitting its .cuda()).
    def proof_scale(s):
        n,k=s.shape; assert n%128==0 and k%4==0
        expected=s.view(torch.uint8).reshape(n//128,4,32,k//4,4).permute(0,3,2,1,4).contiguous().flatten()
        row=torch.arange(n)[:,None]; col=torch.arange(k)[None,:]
        off=(row//128)*(k//4)*512+(col//4)*512+(row%32)*16+((row%128)//32)*4+col%4
        assert off.unique().numel()==n*k
        assert torch.equal(expected[off],s.view(torch.uint8))
        return expected
    rows=[]; scalars=[]
    lut=torch.tensor([0,.5,1,1.5,2,3,4,6,-0.,-.5,-1,-1.5,-2,-3,-4,-6])
    for e in range(256):
        prefix=f'model.language_model.layers.{args.layer}.mlp.experts.{e}.'
        record={'expert':e}; values={}
        for proj,n,k in [('gate_proj',512,2048),('up_proj',512,2048),('down_proj',2048,512)]:
            q=prefix+proj+'.';w=get(q+'weight_packed');s=get(q+'weight_scale');wg=get(q+'weight_global_scale').float().item();ig=get(q+'input_global_scale').float().item()
            assert tuple(w.shape)==(n,k//2) and w.dtype==torch.uint8
            assert tuple(s.shape)==(n,k//16) and s.dtype==torch.float8_e4m3fn
            assert wg>0 and ig>0
            values[proj]=(w,s,wg,ig)
            record[proj]={'packed_sha256':hashlib.sha256(w.numpy().tobytes()).hexdigest(),'linear_scale_sha256':hashlib.sha256(s.view(torch.uint8).numpy().tobytes()).hexdigest(),'weight_global_scale':wg,'input_global_scale':ig}
        g,u,d=[values[z] for z in ['gate_proj','up_proj','down_proj']]
        # Existing helper applies gate global scale to both halves: fail closed if
        # checkpoint differs, rather than silently propagating that assumption.
        assert g[2]==u[2],('gate/up weight global scales differ',e)
        assert g[3]==u[3],('gate/up input scales differ',e)
        w13=torch.cat([u[0],g[0]]);s13=torch.cat([u[1],g[1]])
        sw13=proof_scale(s13);sw2=proof_scale(d[1])
        # Native phase reads up before gate. Check each decoded row against its
        # exact original source; scale reorder proof above is byte-for-byte.
        for offset,raw in [(0,u),(512,g)]:
            packed=w13[offset:offset+512]
            decoded=lut[torch.stack((packed&15,packed>>4),-1).long()].flatten(-2)
            source=lut[torch.stack((raw[0]&15,raw[0]>>4),-1).long()].flatten(-2)
            a=decoded.reshape(512,128,16)*s13[offset:offset+512].float()[...,None]/g[2]
            b=source.reshape(512,128,16)*raw[1].float()[...,None]/raw[2]
            assert torch.equal(a,b)
        record['phase_w13_sha256']=hashlib.sha256(w13.numpy().tobytes()).hexdigest()
        record['phase_sfb_w13_sha256']=hashlib.sha256(sw13.numpy().tobytes()).hexdigest()
        record['phase_sfb_down_sha256']=hashlib.sha256(sw2.numpy().tobytes()).hexdigest()
        rows.append(record);scalars.append((g[3],d[3],g[2],d[2]))
    a1=max(1/x[0] for x in scalars);a2=max(1/x[1] for x in scalars)
    out={'passed':True,'scope':'CPU decoded up/gate and literal phase F8_128x4 scale address proof for all256 experts; no CUDA/captured operand execution','layer':args.layer,'checkpoint':str(args.snapshot),'checkpoint_identity':{f:sha(args.snapshot/f) for f in ['config.json','model.safetensors.index.json']},'effective_input_dequant_scale':a1,'effective_fc2_input_dequant_scale':a2,'adapter_alpha_semantics':'For unchanged unit-scaled SFA from ops.scaled_fp4_quant, use prepared g1=(weight_global_scale reciprocal)*a1; differs from upstream front-end comment about folded SFA','adapter_global_scale_semantics':'Upstream quantize_block_fp4 takes a multiplier: pass prepared a2_gscale=1/a2, NOT installed reciprocal-style fc2_input_scale' ,'adapter_down_alpha_semantics':'Reuse prepared (down weight_global_scale reciprocal)*a2','experts':rows,'cuda_initialized':torch.cuda.is_initialized()}
    assert not out['cuda_initialized'];args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='experts'}))
if __name__=='__main__':main()
