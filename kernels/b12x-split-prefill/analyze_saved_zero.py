"""CPU-only classification limits for the saved initial domain."""
import json,sys
from pathlib import Path
import torch
p=Path(sys.argv[1]);d=torch.load(p/'initial_domain.pt',map_location='cpu',weights_only=True);report=json.loads((p/'report.json').read_text());r=report['cases'][0]['reference']
rows=d['token_map'].numel();b=d['intermediate'].view(torch.uint8);q=b[:rows*256].reshape(rows,256);sf=b[rows*256:].reshape(4,rows,8).permute(1,0,2).contiguous().reshape(rows,32)
valid=d['token_map']>=0;q=q[valid];sf=sf[valid];lo=q&15;hi=q>>4;zero_sign=(lo==8)|(hi==8);zero_scale=(sf==0).repeat_interleave(8,dim=1)
out=dict(valid_rows=int(valid.sum()),actual_payload_bytes=q.numel(),actual_bytes_containing_negative_zero=int(zero_sign.sum()),actual_negative_zero_nibbles=int((lo==8).sum()+(hi==8).sum()),zero_scale_bytes=int(zero_scale.sum()),zero_scale_nonzero_payload_bytes=int(((q!=0)&zero_scale).sum()),actual_scales_all_finite_positive=bool(torch.isfinite(sf.view(torch.float8_e4m3fn).float()).all() and (sf.view(torch.float8_e4m3fn).float()>0).all()),reported_payload_byte_mismatch=r['phase1_payload_byte_mismatch'],reported_scale_byte_mismatch=r['phase1_scale_byte_mismatch'],reported_decoded_value_mismatch=r['phase1_vs_fp64_bf16_requant']['mismatch_count'],limitation='Expected quantized reference payload was not saved. These actual-byte counts cannot establish pairwise mismatch classification. Future numerical/timing precheck will count mismatches after signed-zero canonicalization and isolate zero-scale blocks.',cuda_initialized=torch.cuda.is_initialized())
assert not out['cuda_initialized'];print(json.dumps(out,indent=2))
