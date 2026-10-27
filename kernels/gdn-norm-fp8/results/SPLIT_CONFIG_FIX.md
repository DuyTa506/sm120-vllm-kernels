# Preserve completed baseline split configuration

The first GDN integration supplied 17 `_attention_ops` plus the new GDN op.
That explicit list bypassed vLLM's default completion, which normally adds
`vllm::unified_kv_cache_update` and `vllm::unified_mla_kv_cache_update`.
The installed `enable_rope_kvcache_mla_fusion` predicate consequently became
True, whereas the ordinary next/scalar baseline resolves it to False.
The first startup is not a controlled configuration comparison.

The corrected generator emits the completed baseline19, in their original
order, followed by the single new GDN op (20 total), and explicitly includes
`pass_config.fuse_rope_kvcache_cat_mla=False`. Other supplied JSON fields are
preserved. Explicit base lists missing standard splits, graph partitioning,
or conflicting attention/RoPE fusion settings fail rather than silently
changing their semantics. The overlay independently requires the completed
split set and matching pass settings before enabling.

`audit_integration_config.py` runs these exact installed methods extracted
with AST, without importing Torch/vLLM or probing a platform:

- `CompilationConfig.set_splitting_ops_for_v1`
- `CompilationConfig.splitting_ops_contain_kv_cache_update`
- `enable_rope_kvcache_mla_fusion`

`integration_config_cpu_audit.json` records source hashes, baseline settings,
the first candidate, and corrected settings. It verifies the corrected split
list differs only by the GDN op and the relevant pass settings match the
baseline. This is a characterization of the relevant completion branch using
the observed mode3/no-partition/no-attention-quant inputs, not construction
of the entire platform-dependent VllmConfig. The next startup must confirm
all resolved engine settings against baseline.

The regression failed on the original generator (`split_config_regression_red.log`)
and passes after the correction (`split_config_regression_green.log`).
`integration_compilation_config.json` contains the emitted20-split manifest.
The complete relevant CPU suite and current source hashes are recorded in
`integration_cpu_preflight.json`. The first failed source snapshot remains
unchanged under `results/qwen_sequential/gdn_integration_1`.
