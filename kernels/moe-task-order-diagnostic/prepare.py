"""Copy exact native sources and add a diagnostic store before unchanged atomics."""
import ast, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'moe-task-order'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((SOURCE/'source_manifest.json').read_text())
records={}
for variant,key in [('baseline','baseline_sha256'),('long_first','candidate_sha256')]:
    source=SOURCE/f'gated_{variant}.py'
    assert sha(source)==manifest[key]
    s=source.read_text()
    start=s.index('@dsl_user_op\ndef scatter_add_weighted_bf16x8_packed_alpha(')
    end=s.index('\n\nclass DynamicLaunchParams',start)
    helper=s[start:end]
    helper=helper.replace('    down_alpha,\n','    down_alpha,\n    log_addr, meta_addr, token, expert, slice_begin, slice_count,\n')
    helper=helper.replace('            down_alpha.ir_value(loc=loc, ip=ip),','            down_alpha.ir_value(loc=loc, ip=ip),\n'+''.join(f'            {typ}({name}).ir_value(loc=loc, ip=ip),\n' for name,typ in [('log_addr','Int64'),('meta_addr','Int64'),('token','Int32'),('expert','Int32'),('slice_begin','Int32'),('slice_count','Int32')]))
    helper=helper.replace('        " createpolicy.fractional.L2::evict_last.b64 cp, 1.0;"','        " .reg .pred logp, metap;"\n        " setp.ne.u64 logp, $4, 0;"\n        " @logp st.global.v4.u32 [$4], {p0,p1,p2,p3};"\n        " setp.ne.u64 metap, $5, 0;"\n        " @metap st.global.v4.u32 [$5], {$6,$7,$8,$9};"\n        " createpolicy.fractional.L2::evict_last.b64 cp, 1.0;"')
    helper=helper.replace('"l,r,f,f",','"l,r,f,f,l,l,r,r,r,r",')
    s=s[:start]+helper+s[end:]
    start=s.index('    def scatter_sC_to_gmem(')
    end=s.index('    @cute.jit\n    def load_fc1_tma_slice(',start)
    method=s[start:end].replace('        down_alpha_value,\n','        down_alpha_value,\n        task_m_tile_idx, task_expert_idx, task_slice_begin_idx, task_slice_count_val,\n',1)
    # Every physical tile has slots for two split groups; unsplit uses group zero.
    method=method.replace('        epi_rest_m =','        log_slot = task_m_tile_idx * Int32(2) + task_slice_begin_idx // Int32(2)\n        epi_rest_m =',1)
    lines=method.splitlines(); out=[]
    for line in lines:
        if line.strip()=='scatter_add_weighted_bf16x8_packed_alpha(':
            indent=line[:len(line)-len(line.lstrip())]
            out += [indent+'log_addr = Int64(0)',indent+'meta_addr = Int64(0)',indent+'if output_tile_idx < Int32(self.diag_columns // 128):',indent+'    log_row = log_slot * Int32(128) + cached_row',indent+'    log_addr = Int64(self.diag_payload_ptr) + Int64(log_row * Int32(self.diag_columns) + global_col) * Int64(2)',indent+'    if global_col == Int32(0):',indent+'        meta_addr = Int64(self.diag_meta_ptr) + Int64(log_row) * Int64(16)']
        out.append(line)
        if line.strip()=='down_alpha_value,' and len(line)-len(line.lstrip())>8:
            out.append(line[:len(line)-len(line.lstrip())]+'log_addr, meta_addr, tok, task_expert_idx, task_slice_begin_idx, task_slice_count_val,')
    s=s[:start]+'\n'.join(out)+'\n'+s[end:]
    marker='                        scatter_weight_base_addr,\n                        down_alpha_value,\n'
    assert s.count(marker)==1
    s=s.replace(marker,marker+'                        task_m_tile_idx, task_expert_idx, task_slice_begin_idx, task_slice_count_val,\n')
    ast.parse(s)
    dest=HERE/f'gated_{variant}_diagnostic.py'; dest.write_text(s)
    records[variant]={'source':str(source),'source_sha256':sha(source),'diagnostic_sha256':sha(dest)}
(HERE/'source_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
