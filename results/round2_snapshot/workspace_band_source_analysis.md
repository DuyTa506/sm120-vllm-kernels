# Workspace band screen and remaining native mechanism

`results/workspace_bands_smoke_1/screen_summary.json` records the first complete scalar-call graph screen on captured L19 inputs. M1024/tile32 regressed 1.89–1.91%; M2048/tile64 regressed 27.05–27.06%; unchanged M4096/tile128 was within −0.32…−0.12%. Three bands allocated 1,400,907,776 bytes versus 607,809,536 bytes for the parent. Six graph band transitions passed; diagnostic numerical checks passed, but no production quality equivalence is established. This workspace policy is not selected for serving. No repeat or serving test is justified by this negative screen alone.

## Inactive M16 MMA fragments are already skipped

The authoritative source below is the actual installed baseline saved with the smoke:
`results/workspace_bands_smoke_1/runtime_sources/0_gated.py`, SHA256
`993783308233288ddfa77293e9dbabdc825ba5bfdcc4dcc41e842a895ec33445`.

- Lines 4458–4464 compute `valid_rows` and `warp_m_coord = warp_idx & 3` for atom layout `(4,2,1)`. A warp owns two M16 fragments separated by 64 rows at M128.
- Lines 4499–4589 select the branch-free full FC1 path only for full tiles; all tails call `fc1_gate_up_swiglu_to_sC_tail`.
- Tail FC1 MMA guards at lines 1848, 1913 and 2014 test `valid_rows > mt*64 + warp_m_coord*16`. This covers gate and up operations. Activation/held-accumulator stores are similarly guarded at 2050 and 2061.
- FC2 selects its tail path at 4705–4732. Its MMA guard at 2524 uses the same M16 condition.
- Q1 quantization and scatter already honor valid rows (2129, 2264, 2298, 2612).

Thus reducing M128 padding does **not** correspond to removing full padded tensor-core work. The requested inactive-M16 MMA mechanism is already present; adding it again is not a new optimization. The timing result is consistent with additional tile/task/weight overhead, but does not isolate the cause.

For the separate saved L19 geometry histograms (`task_order_geometry_1`, not the smoke's prefix distributions):

| Tokens | M128 tiles | Padded rows | Rows rounded to active M16 | Completely inactive MMA warp assignments |
|---|---:|---:|---:|---:|
| 2048 | 274 | 35072 | 18192 | 596 / 2192 |
| 3072 | 339 | 43392 | 26400 | 594 / 2712 |
| 4096 | 400 | 51200 | 34528 | 574 / 3200 |

These are static counts before weighting by split slice count, not instruction/traffic savings predictions.

## Concrete unimplemented candidate: skip operand register loads in fully inactive warps

The current tail methods still load operands from shared memory even when a warp has no active M16 fragment at all:

- FC1 initial A/B/up-B/SFA/SFB register loads at 1787–1819 and subsequent K-block loads remain outside the MMA predicate.
- FC2 B/SFB loads at 2481–2522 remain outside its predicate.
- `load_fc2_a_fragments` at 2322–2363 loads A/SFA unconditionally; call site 4696 is inside every output tile and slice iteration. Its comment about reuse across all output tiles overstates the current call placement. Multi-slice tasks overwrite these registers, so simply hoisting the call is incorrect.

A narrow M128-tail experiment can use warp-uniform `valid_rows > warp_m_coord*16` to guard **only** shared-to-register operand copies for entirely inactive warps. Leave every pipeline wait/release, phase-state advance, named barrier, producer TMA transaction and quantization/scatter loop unchanged. Do not predicate ldmatrix per lane: the proposed condition is uniform across all 32 lanes. Do not skip the whole consumer warp, because its quantization duties and synchronization contributions remain necessary. Keep accumulator zero initialization and epilogue stores unchanged initially.

This preserves task count, weight TMA traffic, M128 workspace, MMA layout and array declarations; it targets redundant LDS/ldmatrix and scale-register loads, not global weight bandwidth or additional MMA elimination. It needs an extra predicate but no new fragment arrays. Actual register count, spills and SASS must still be checked: unchanged declarations do not guarantee unchanged compiled register pressure. The producer can still determine the critical path, so benefit may be zero or negative.

Safety boundaries: FC1 pipeline waits/releases at 1826–1844 and FC2 waits/releases at 2473–2474/2505–2506 must stay unconditional. Eight MMA warps remain participants in the 256-thread epilogue barrier; FC2 group barriers retain 128 threads and pass barriers retain all 288 threads (797–818). Existing proxy fences and Q1 publication barriers at 4595–4596 and 4635–4637 remain intact. No change to SFA storage/publication is proposed, only skipping register reads in a warp that will not consume them.

No kernel implementation or GPU run has been performed for this candidate. A reviewed experiment should instantiate only the exact current M128/I512/K2048 path, compare fixed native source/config and full-call graph span, preserve changed-input/routes checks, and inspect generated load instructions and register counts before considering serving.

## Split scheduling alternative

Current publisher already splits N slices to target approximately four resident waves (3581–3599), and `publish_variable_deferred_tasks` assigns split status by global M-tile index (1084). Changing only queue ordering would duplicate the tested long-first idea. A truly different policy could choose which M tiles receive extra N splitting using row-dependent estimated work, but it changes duplicated phase/task overhead and reduction order. There is no calibrated cost model or positive evidence yet; this is lower priority than the narrowly scoped inactive-operand-load test.
