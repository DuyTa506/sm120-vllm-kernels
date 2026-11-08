# Native NVFP4 follow-up hypothesis (CPU audit only)

Priority remains validating the fused FP8 QKVZ candidate on captured operands. This note identifies a small native NVFP4 experiment without claiming a DeepGEMM MXFP4 proxy is NVFP4.

## Actual source boundary

The installed FlashInfer source in `dg-sm120-env:1` is `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/_moe_dynamic/gated.py`. CPU inspection shows `_SF_VEC_SIZE=16`, `MmaMXF4NVF4Op` and blockscale layouts selected by `sf_vec_size`, plus original expert/global scaling. Retaining this path preserves its native NVFP4 representation and packed-A permutation.

In contrast, the local DeepGEMM `mma/sm120.cuh` FP4 instruction explicitly uses `scale_vec::2X` with UE8M0 and two scale bytes per K64. Supporting per16 E4M3 requires the corresponding four-scale fragment, scale loads/layout, matching MMA mode and host validation; changing the public scale granularity flag alone is invalid.

## Small experiment: long tasks before short tasks

The installed b12x dynamic kernel already has a compact packed descriptor queue, atomic claims/prefetch and split-task heuristics. For 256<M<=4096, it splits a prefix of physical M tiles into two-slice tasks to reach roughly four resident waves. `publish_variable_deferred_tasks` currently queues that short-task prefix before remaining longer full-slice tasks. Therefore merely adding a dynamic scheduler duplicates existing behavior.

Hypothesis: enqueue unchanged unsplit (long) descriptors first, followed by unchanged split (short) descriptors to reduce final-wave imbalance. Keep physical tile indices, routing top-k, packed activation bytes/scales, expert weights, FC1/FC2 math and each task's local accumulation order unchanged. For split_count S, total physical tiles T and G split groups, the alternative slots are:

- unsplit tile t>=S: `slot=t-S`;
- split tile t<S, group g: `slot=(T-S)+t*G+g`.

This requires passing T to the publisher and changing slot calculation, not a new permutation kernel or router. Start Qwen E256/top8/K2048/intermediate512 at M1024/2048, using actual captured routing plus uniform/hot-expert controls. Record S/T/G and task-count tail statistics to verify the hypothesis actually applies. Measure the complete native call, including queue publication.

## Numerical constraint

Current b12x scatter calls `red.global.add.noftz.v4.bf16x2`; task execution order can change global BF16 reduction order. The proposed queue ordering therefore **cannot claim bitwise-preserved finalize accumulation**, even though descriptor math is unchanged. A strict unchanged-order requirement needs a per-route scratch plus fixed-order finalize path, which is a larger experiment and adds traffic. Otherwise, baseline self-repeat variation and model NLL gates must be assessed explicitly before any promotion. No source patch or GPU experiment for this hypothesis has been run.
