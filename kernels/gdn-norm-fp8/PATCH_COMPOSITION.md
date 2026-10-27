# Experimental scalar + GDN patch composition

These patches reproduce the frozen scalar MoE and GDN overlay sources. They
are **experimental artifacts, not a selected or recommended configuration**.
The full `scalar_gdn_matched_3` comparison was still running when this packaging
check was prepared. Source identity is not a combined quality or speed claim.

| Patch | Required baseline | Result |
|---|---|---|
| [`06-moe-b12x-scalar-input-scale.patch`](../moe-nvfp4/06-moe-b12x-scalar-input-scale.patch) | `shared_exact` MoE file, after MoE patches 03 → 04 → 05 | Exact `scalar/` MoE file; adds only the opt-in scalar flag and load-time hook |
| [`01-gdn-norm-fp8-experimental.patch`](01-gdn-norm-fp8-experimental.patch) | Existing `CONFIG=next` GDN file from `build/`, after root vLLM patches 01–05 | Exact five-file GDN overlay: changes `qwen_gdn_linear_attn.py` and adds four helpers |

Both use package paths rooted at `vllm/`. MoE patch 03 also changes
`fused_moe/oracle/unquantized.py`; retain that prerequisite for the existing MTP
fallback. The GDN baseline depends on the root 02 internal-checkpoint and 05
fused-convolution changes, including the existing `fused_conv_post_conv.py` helper.
These new patches do not replace those prerequisites.

Once both baselines are present, the new scalar and GDN patches touch disjoint
paths and apply in either order. They remain outside
`patches/vllm-v0.32.0/*.patch`, so the default build script does not autoapply them.
No runtime source, build output, harness, default `CONFIG=next` behavior, or
installed package was changed during packaging.

## Reproduction evidence

Run from the repository worktree:

```sh
python3 -I kernels/gdn-norm-fp8/verify_patch_composition_cpu.py \
  --output kernels/gdn-norm-fp8/results/patch_composition_cpu.json
```

The verifier imports only Python standard-library modules and runs GNU `patch`
only in disposable temporary directories. It checks the pinned source manifests,
uses `--fuzz=0`, rejects offset/fuzz application, checks Python syntax without
imports, and proves:

- Saved pristine MoE files plus patches 03 → 04 → 05 → 06 reproduce each
  intermediate `patched`, `shared`, `shared_exact`, and `scalar` backend, while
  retaining the expected patched oracle.
- Fresh `shared_exact` + next-GDN trees produce all six exact overlay files when
  applying scalar then GDN, and when applying GDN then scalar.
- Source hashes match completed scalar/exact/integration manifests and the
  `scalar_gdn_matched_3` startup source capture. The latter is an identity check,
  not evidence that its full run completed or passed.
- The source overlays and both starting baseline files remain unchanged.

The [JSON manifest](results/patch_composition_cpu.json) records every patch hash,
baseline/output SHA256, measured source-manifest hash, application log, and
per-file result. The two modified-file baseline hashes are:

| Baseline | SHA256 |
|---|---|
| `shared_exact/.../flashinfer_b12x_moe.py` | `8c703524b0331521c0785d24677cecc03f08d0e41cdbe159193acf6123752e86` |
| `build/.../qwen_gdn_linear_attn.py` | `c82a390240c77032414d916ed705780716cd696f90722eb99eb89419f0aa262c` |

Do not fuzz these patches onto a changed baseline. Rebase and regenerate the
patches if an upstream source, another overlay on either target file, or any of
the four newly added GDN helper names conflicts. Applying the scalar patch
directly to `shared/` could omit the exact CUTLASS alpha correction even if its
local hunks apply; the full-file baseline hash is therefore part of the contract.

## Runtime opt-in contract

Packaging does not enable either experiment. Use the existing isolated runner's
`scalar_gdn` variant for a scheduled test, preserving the pinned vLLM 0.32.0 /
FlashInfer / Torch environment and the existing `CONFIG=next` flags. The explicit
additional settings are:

- `VLLM_B12X_FIX_SCALES=1`, `VLLM_B12X_SHARED_WORKSPACE=1`, and
  `VLLM_B12X_SCALAR_INPUT_SCALE=1`, with the existing static activation-scale
  convention and hybrid threshold 1024. The scalar hook requires the targeted
  Qwen MoE shape and exactly uniform, positive, finite FC1 scales; unsupported
  cases retain the vector scale. It changes no forward-time scale checks.
- All five GDN files must be present together, with
  `VLLM_GDN_NORM_FP8_FUSION=1`. GDN eligibility remains guarded: SM120, TP/PP1,
  supported compiled Inductor/packed path, fixed MTP capture coverage, and the
  original dynamic-token CUTLASS FP8 consumer. Post-load validation/warmup and
  all 30 readiness records remain required.
- Before constructing `VllmConfig`, use the existing
  [`integration_config.py`](integration_config.py) merge. Preserve all 19
  completed baseline splitting ops and add
  `vllm::gdn_attention_core_norm_fp8_packed_experimental` as the twentieth;
  preserve `pass_config.fuse_rope_kvcache_cat_mla=False`. The patch alone cannot
  supply this CLI/configuration requirement. Its helper hash is in the manifest.

Both flags default off in their source files. Retain separate compilation caches
and the same FI control contract on both sides of any comparison; these patches
do not include or modify the FI importer. Read
[`INTEGRATION_PREP.md`](INTEGRATION_PREP.md) for fallback, state, padding and
warmup constraints, and
[`SCALAR_DISPATCH.md`](../moe-nvfp4/SCALAR_DISPATCH.md) for scalar eligibility and
validation scope. The existing `py_email30k` NLL flag remains open; packaging
does not resolve it. Final adoption/composition requires the pending paired
serving and quality evidence.
