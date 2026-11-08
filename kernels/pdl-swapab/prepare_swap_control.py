#!/usr/bin/env python3
"""Prepare a reviewable native-SM120 swap override patch; no build or source mutation."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

OLD='''    const bool swap_ab = (m >= 1 and m <= kSwapAbMMax
        and d.stride(-1) == 1 and !is_mixed_fp4 and !c.has_value());'''
NEW='''    const bool swap_eligible = (m >= 1 and m <= kSwapAbMMax
        and d.stride(-1) == 1 and !is_mixed_fp4 and !c.has_value());
    const char* swap_control = std::getenv("DG_SM120_SWAP_AB_CONTROL");
    DG_HOST_ASSERT(swap_control == nullptr or std::strcmp(swap_control, "auto") == 0
        or std::strcmp(swap_control, "off") == 0 or std::strcmp(swap_control, "on") == 0);
    const bool force_on = swap_control != nullptr and std::strcmp(swap_control, "on") == 0;
    const bool force_off = swap_control != nullptr and std::strcmp(swap_control, "off") == 0;
    DG_HOST_ASSERT(not force_on or swap_eligible);
    const bool swap_ab = force_on or (swap_eligible and not force_off);
    if (std::getenv("DG_SM120_SWAP_AB_TRACE") != nullptr) {
        std::printf("SWAP_AB_CONTROL_V1 m=%d n=%d k=%d eligible=%d selected=%d\\n",
                    m, n, k, int(swap_eligible), int(swap_ab));
        std::fflush(stdout);
    }'''

def prepare(source):
    if source.count(OLD)!=1:raise ValueError('native swap gate source mismatch')
    if source.count('#pragma once')!=1:raise ValueError('header guard mismatch')
    return source.replace('#pragma once','#pragma once\n#include <cstdlib>\n#include <cstring>\n#include <cstdio>',1).replace(OLD,NEW,1)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'swap_control');a=p.parse_args()
    relative=Path('csrc/apis/sm120_gemm.hpp');source=(a.source_root/relative).read_text();candidate=prepare(source)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    diff=''.join(difflib.unified_diff(source.splitlines(True),candidate.splitlines(True),fromfile='a/'+str(relative),tofile='b/'+str(relative)))
    (a.output_dir/'native_swap_override.patch').write_text(diff)
    sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
    manifest={'status':'CPU_prepared_not_built','source_root':str(a.source_root.resolve()),'file':str(relative),'base_sha256':sha(source),'patched_sha256':sha(candidate),'patch_sha256':sha(diff),
     'scope':'Native dense same-dtype FP8 or packed FP4, original eligible M1..16 only, row-contiguous output, no accumulation. No rowwise adapter or NVFP4 format conversion.',
     'controls':['auto','off','on'],'unsupported_force_on':'Fails original eligibility gate; never silently falls back.'}
    (a.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
