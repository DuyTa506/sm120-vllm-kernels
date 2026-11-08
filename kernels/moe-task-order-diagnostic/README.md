# Native NVFP4 task-order numerical diagnostic

Both tile-zero and full2048-column baseline/candidate diagnostics completed successfully. This is numerical evidence, not a benchmark or a serving patch. Owned separately from `../moe-task-order`; detailed results and limitations are below.

The long-first L19/M4096 experiment changes BF16 atomic summation order. Its cross-variant drift (~0.00482 relative L2) exceeds baseline repeat drift (~0.003). This diagnostic distinguishes changes in individual contributions from changes in their global reduction. It uses the unchanged saved activation, expert IDs and routing weights from `task_order_4096_1/measurement/a1/l19_m4096_real.pt`, and the real checkpoint weights.

`prepare.py` verifies source hashes against the native experiment manifest and copies both variants. Only the scatter helper and its argument plumbing are changed: after the exact original BF16x2 route-weight multiplication, a predicated `st.global.v4.u32` records those registers before the original BF16 atomic add. The original atomic output remains intact. Logging is enabled only for physical FC2 N tile zero (128 columns), even when output tiles are visited in expert-rotated order. The FC1/FC2 math, descriptors, producer scheduling and atomic instruction are unchanged in source. Extra stores can perturb execution timing, so this cannot establish performance or perfectly reproduce the uninstrumented atomic interleaving.

L19/M4096 has 400 physical M tiles, 335 split tiles and two groups, totaling 735 live descriptors. The scratch uses 800 slots (`physical_tile*2 + slice_begin//2`), including unused group slots. BF16 payload is 800×128×128×2 = 26,214,400 bytes (25 MiB); four-int metadata per row is 1,638,400 bytes (1.5625 MiB). Full 2048-column BF16 logging would use 400 MiB, versus 367.5 MiB for densely packed 735 descriptors; it fits memory but adds unnecessary stores. The tile-zero subset is the first discriminator. Existing epilogue has no non-atomic scratch-output interface, hence the isolated diagnostic helper.

Metadata contains token ID, expert, slice start and slice count. Only the unique local-column-zero lane writes row metadata; all 16 disjoint N8 vectors write payload. Inactive slots/tail rows initialize to NaN/-1 and must remain untouched. The CPU comparator rejects duplicate/missing canonical contributions, incomplete columns, wrong physical-slot ownership, unexpected writes and nonfinite values. It canonicalizes by `(token, expert, slice_begin, slice_count)` to remove Q0's nondeterministic physical row permutation. The observed split boundary starts expert230 exactly; the comparator rejects an expert straddling the split boundary, where routes alone could not reconstruct each token's split class.

The comparator sums exact logged BF16 contributions in fixed canonical order with FP64 accumulation. It reports bitwise A/B payload mismatch, reference mismatch, each atomic output's error versus its own fixed reference, and cross-atomic error. Equal payload/reference with differing final outputs isolates reduction order for these columns. Differing payloads do not justify that conclusion; the next probe would inspect pre-BF16 FC2 accumulators/Q1 bytes. This first diagnostic records the exact payload presented to atomics, not the preconversion FP32 GEMM accumulators.

CPU checks passed:

```bash
python3 kernels/moe-task-order-diagnostic/prepare.py
python3 -m py_compile kernels/moe-task-order-diagnostic/*.py
# In the existing container, with CUDA_VISIBLE_DEVICES empty:
python3 /work/kernels/moe-task-order-diagnostic/import_preflight.py
python3 /work/kernels/moe-task-order-diagnostic/test_compare.py
```

GPU execution requires the root agent's explicit exclusive slot and owned-process launcher. Run variants in separate processes/caches; retained scratch pointers are embedded in each diagnostic kernel object and cannot be reused with another allocation. Proposed inner command (not executed):

```bash
python3 /work/kernels/moe-task-order-diagnostic/diagnose.py \
  --variant baseline \
  --capture /work/kernels/moe-task-order/results/task_order_4096_1/measurement/a1/l19_m4096_real.pt \
  --output /work/kernels/moe-task-order-diagnostic/results/UNIQUE/a
```

Then a separately authorized candidate call and CPU `compare.py A/diagnostic.pt B/diagnostic.pt --output comparison.json`. No timers are collected. GPU JIT correctness remains unverified until the granted run.

The owned launcher is now prepared and CPU preflight passed:

```bash
sg docker -c 'bash kernels/moe-task-order-diagnostic/run.sh task_order_diag_a1 baseline'
```

Do not execute without the explicit GPU slot. Each variant performs two observed calls after warmup; outputs are `diagnostic.pt` and `diagnostic_repeat1.pt`. Compare baseline repeat-to-repeat as well as baseline-to-candidate. The launcher acquires the shared exclusive lock, requires an empty GPU, snapshots diagnostic and runtime sources, records image identity, preserves exit status, verifies original source hashes afterward, and cleans only its recorded subprocess group. Runtime scratch pointers stay alive until all calls synchronize. Constructor tile geometry and checkpoint packed dimensions are checked before warmup; T400/S335/G2 is checked before observed calls. Checkpoint identity records snapshot path plus config/index SHA, not full weight-shard hashes.

## Tile-zero result

Both `task_order_diag_a1` and `task_order_diag_b1` exited0 with source verification and owned-process cleanup. Two observed calls per variant each covered 59,198 canonical contributions. Independent all-pairs analysis in `results/task_order_diag_pair1.json` found zero BF16 payload-bit mismatches and zero fixed-FP64 reference differences for A/A, B/B and all four A/B pairings. Final atomic outputs still differed: A/A relative L2 0.003300, B/B 0.002893, A/B 0.004778–0.004851. Error against the shared fixed-FP64 sum was 0.004424–0.004444 for A and 0.004067–0.004081 for B.

This isolates accumulation order as the source of observed differences for the logged first 128 output columns and these actual L19/M4096 operands. Candidate error happened to be smaller against this reference; that is not a general accuracy improvement or permission to promote the serving patch. Instrumentation changes timing, and the remaining columns were not observed in this first pair.

## Full-width option (CPU prepared)

The default remains 128 logged columns. `diagnose.py --columns 2048`, or launcher third argument `2048`, records all 16 physical FC2 N tiles. Actual output width is guarded as 2048 before launch. Payload addresses use the global column and selected row stride; metadata remains written exactly once at physical N tile zero/local column zero. NaN/-1 sentinels and canonical comparison infer and verify either supported width, including the final column.

Full-width payload is 800×128×2048×2 = 419,430,400 bytes (400 MiB), plus 1.5625 MiB metadata. Two calls per variant are retained; file/storage and CPU analysis costs grow accordingly. This is still an isolated numerical diagnostic without performance claims. Full-width GPU compilation/execution subsequently completed; see the full2048 result below.

```bash
# Only after explicit exclusive GPU grant; one variant per invocation.
sg docker -c 'bash kernels/moe-task-order-diagnostic/run.sh task_order_diag_full_a1 baseline 2048'
```

## Full2048 result

`task_order_diag_full_a1` and `task_order_diag_full_b1` completed with exit0. Independent `results/task_order_diag_full_pair1.json` compares both repetitions of each variant, including all four cross pairings. Each call contains 59,198 canonical contributions over **all 2048 output columns**. Every payload-bit comparison and fixed-FP64 reference comparison is exactly equal. Atomic output relative L2 is 0.003261 for A/A, 0.003278 for B/B, and 0.004748–0.004752 for A/B. Relative error against the common FP64 sum is 0.004354–0.004372 for A and 0.004034–0.004038 for B.

This extends the earlier tile-zero conclusion to the full FC2 output for the actual captured L19/M4096 operands: observed drift comes from BF16 accumulation order while the weighted local contributions are unchanged. It remains a diagnostic of this layer/input, not model-quality promotion. An uninstrumented opt-in serving screen is being prepared separately in `../moe-task-order-serving`; no default backend/router change follows from this result.
