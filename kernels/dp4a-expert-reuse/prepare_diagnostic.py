from pathlib import Path
p=Path(__file__).resolve().parent;s=(p/'qf_moe_port.cu').read_text()
old='        float v[16];\n#pragma unroll\n        for (int j = 0; j < 16; ++j) v[j] = s_act[rk][j];'
assert s.count(old)==1
s=s.replace(old,'''        // Diagnostic-only stage snapshots: pre-BF16 FC1 up/gate and BF16 SwiGLU.
        float* debug = hsf + MAXSLOT * MAXT * IB + ((size_t)pos * MAXT + t) * (3*I);
        for (int j=0;j<16;++j) {
            debug[jt*16+j]=s_dot[j][rk];
            debug[I+jt*16+j]=s_dot[16+j][rk];
            debug[2*I+jt*16+j]=s_act[rk][j];
        }
'''+old)
s=s.replace('hsf.numel()>=MAXSLOT*MAXT*IB','hsf.numel()>=MAXSLOT*MAXT*(IB+3*I)')
s=s[:s.index('PYBIND11_MODULE')]+'''
// Independent quant16 probe uses the exact compiled device helper of this port.
__global__ void quant_probe_kernel(const float* x,float gs,uint32_t* out,float* sf,int blocks) {
 int b=blockIdx.x*blockDim.x+threadIdx.x;
 if(b<blocks) { float v[16]; for(int j=0;j<16;++j)v[j]=x[b*16+j]; float f;
 uint2 q=quant16(v,gs,f); out[b*2]=q.x;out[b*2+1]=q.y;sf[b]=f; }
}
void quant_probe(at::Tensor x,double gs,at::Tensor out,at::Tensor sf) {
 TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kFloat && x.is_contiguous() && x.numel()%16==0);
 TORCH_CHECK(out.is_cuda() && out.device()==x.device() && out.scalar_type()==at::kInt && out.is_contiguous() && out.numel()==x.numel()/8);
 TORCH_CHECK(sf.is_cuda() && sf.device()==x.device() && sf.scalar_type()==at::kFloat && sf.is_contiguous() && sf.numel()==x.numel()/16);
 const c10::cuda::CUDAGuard guard(x.device());
 int blocks=x.numel()/16;
 quant_probe_kernel<<<(blocks+127)/128,128,0,c10::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),gs,(uint32_t*)out.data_ptr<int>(),sf.data_ptr<float>(),blocks);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("apply", &nvfp4_moe_decode); m.def("quant_probe",&quant_probe); }
'''
(p/'qf_moe_diag.cu').write_text(s)
