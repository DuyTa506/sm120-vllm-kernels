#!/usr/bin/env python3
"""Add a narrow validated tiny-QKVZ policy only to the owned copied source."""
import difflib,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;owned=HERE/'rowwise_tile_control';root=owned/'src'
sha=lambda b:hashlib.sha256(b).hexdigest()
base=json.loads((owned/'base_fingerprint.json').read_text())['actual_files_sha256']
rel='csrc/jit_kernels/impls/sm120_fp8_fp4_gemm_1d1d.hpp';path=root/rel;old=path.read_text()
assert sha(old.encode())==base[rel]
needle='''    auto config = get_best_sm120_config<SM120ArchSpec>(desc);
    config.split_k_factor = SM120ArchSpec::get_split_k_factor(desc, config.layout);
    DG_HOST_ASSERT(row_scale.has_value() == col_scale.has_value());'''
assert old.count(needle)==1
replacement='''    auto config = get_best_sm120_config<SM120ArchSpec>(desc);
    const char* rowwise_tile = std::getenv("DG_ROWWISE_SWAP_TILE");
    const bool narrow = rowwise_tile and std::strcmp(rowwise_tile, "64x16") == 0;
    DG_HOST_ASSERT(not rowwise_tile or std::strcmp(rowwise_tile, "auto") == 0 or narrow);
    if (narrow) {
        DG_HOST_ASSERT(row_scale.has_value() and col_scale.has_value() and swap_ab);
        DG_HOST_ASSERT(m == 12288 and n >= 1 and n <= 16 and k == 2048);
        DG_HOST_ASSERT(not c.has_value() and not alpha.has_value());
        const auto candidates = SM120ArchSpec::get_layout_candidates(desc);
        int matches = 0;
        for (const auto& layout : candidates) {
            if (layout.block_m == 64 and layout.block_n == 16 and layout.block_k == 64) {
                config.layout = layout;
                ++ matches;
            }
        }
        DG_HOST_ASSERT(matches == 1); // Fail if native legality/selection changes.
        config.storage_config = SM120ArchSpec::get_storage_config(desc, config.layout);
        config.pipeline_config = SM120ArchSpec::get_pipeline_config(desc, config.layout, config.storage_config);
        config.launch_config = SM120ArchSpec::get_launch_config(desc, config.layout);
    }
    config.split_k_factor = SM120ArchSpec::get_split_k_factor(desc, config.layout);
    DG_HOST_ASSERT(row_scale.has_value() == col_scale.has_value());'''
new=old.replace(needle,replacement)
needle='''        DG_HOST_ASSERT(not alpha.has_value() and not epilogue_type.has_value());
    }

    const auto cd = c.value_or(d);'''
assert new.count(needle)==1
replacement='''        DG_HOST_ASSERT(not alpha.has_value() and not epilogue_type.has_value());
        if (std::getenv("DG_ROWWISE_SWAP_TRACE")) {
            const int logical_tiles = ceil_div(m, config.layout.block_m) * ceil_div(n, config.layout.block_n);
            std::printf("ROWWISE_POLICY_V1 policy=%s effective_m=%d effective_n=%d k=%d block_m=%d block_n=%d block_k=%d logical_tiles=%d grid_ctas=%d first_wave_work=%d smem=%d stages=%d split_k=%d\\n",
                narrow ? "64x16" : (swap_ab ? "auto_swap" : "off"), m, n, k,
                config.layout.block_m, config.layout.block_n, config.layout.block_k,
                logical_tiles, config.launch_config.num_sms, std::min(logical_tiles, config.launch_config.num_sms),
                config.pipeline_config.smem_size, config.pipeline_config.num_stages, config.split_k_factor);
            std::fflush(stdout);
        }
    }

    const auto cd = c.value_or(d);'''
new=new.replace(needle,replacement);new='#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n'+new
path.write_text(new);patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
(owned/'rowwise_tile.patch').write_text(patch)
parent=json.loads((HERE/'rowwise_swap_control/manifest.json').read_text());files=dict(parent['files']);files[rel]={'base_sha256':sha(old.encode()),'patched_sha256':sha(new.encode())}
x={'status':'CPU_prepared_not_built','base_fingerprint_sha256':sha((owned/'base_fingerprint.json').read_bytes()),'files':files,'patch_sha256':sha(patch.encode()),'scope':'Explicit 64x16x64 alternative for swapped effective12288xM x2048, M1..16. AUTO/off remain separate; no splitK/noC/scale-order changes.'}
(owned/'manifest.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
