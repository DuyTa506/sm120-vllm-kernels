# First OFF/ON screen: rejected

Both `inactive_load_off_1` and `inactive_load_on_1` exited0 with exact child-group cleanup, empty GPU, unchanged14-source snapshots and unchanged13 runtime/helper source hashes. Captured operands and checkpoint identity match. Full runtime JSON differs only in the CPU unittest elapsed-time text within `import_check`; `results/runtime_pair_audit.json` preserves that difference. Actual CUDA graph trace names identify the baseline class in OFF and `MoEGatedDynamicKernelInactiveLoads` in ON at both shapes.

| L19 tokens, tile128 | OFF graph span µs, three block medians | ON graph span µs, three block medians |
|---|---|---|
| 1024 | 314.416 / 314.176 / 314.640 | 880.848 / 871.664 / 891.184 |
| 4096 | 437.360 / 436.912 / 438.064 | 1483.024 / 1480.000 / 1486.160 |

The candidate is roughly2.8× and3.4× slower, respectively. This is a single sequential OFF/ON process pair, but the regression is too large to justify promotion or a repeat for selection. No serving or agentic run is warranted for this variant.

All8 paired numerical cases passed the exploratory finite/relL2<=.02 gate; maximum paired relL2 was.00546438. This is not quality equivalence or independent before-atomic contribution validation. Saved input hashes match for real, changed-input/routes/weights, synthetic padded tails and restored scenarios. Detailed comparisons are in `results/pair1.json`.

## Compiled artifact findings

The persisted `.o` is a host ELF containing an embedded NVIDIA executable ELF, so direct `cuobjdump` initially reported no device code. The first exploratory extraction omitted program headers and is preserved as failed evidence. The corrected extraction includes both section and program header tables/segments; its offset, length, outer-object SHA and extracted SHA are recorded in each run's `embedded_full_manifest.json`. No benchmark sources were changed to obtain this derived evidence. Host `nvdisasm` produced complete SASS after the container's missing executable prevented disassembly.

| Static compiled evidence | OFF | ON |
|---|---:|---:|
| Embedded cubin bytes | 291680 | 474320 |
| Resource REG | 168 | 168 |
| Resource STACK bytes | 80 | 688 |
| Resource static SHARED bytes | 1024 | 1024 |
| Resource LOCAL field | 0 | 0 |
| Disassembled instruction sites | 12057 | 15619 |
| LDL sites | 4 | 1484 |
| STL sites | 5 | 1099 |
| LDSM sites | 384 | 384 |
| OMMA sites | 896 | 896 |

`results/sass_pair_summary.json` contains full static opcode counts. These counts are instruction sites, not dynamic instruction counts. The unchanged resource LOCAL field does not demonstrate absence of local-memory traffic: the SASS explicitly shows a large increase in LDL/STL. Likewise SHARED1024 is the static allocation field, not total dynamic shared memory at launch.

The huge local load/store increase and enlarged stack are consistent with compiler spilling/stack traffic introduced by conditional operand definitions. Unchanged register declarations and the same REG168 do not preserve allocation/liveness behavior. This is a credible mechanism for the slowdown, not a measured stall attribution. The proposed warp-uniform copy guards must therefore be rejected in this form, despite keeping algorithm, task queue and synchronization AST unchanged. No patch is added to the serving stack.

## CPU-only repair proposal, not implemented

A plausible source of the allocation change is conditional definition of register-fragment elements: the skipped-copy path carries their old values through a join and potentially across unrolled K iterations. The candidate's large LDL/STL increase supports investigating that value lifetime, although SASS counts alone do not identify the compiler's exact SSA transformation.

For a new, separately named diagnostic variant, first revert all FC1 and common A/SFA guards and isolate the six FC2-tail B/SFB register copies. Immediately before each guarded copy, unconditionally zero only its destination register slice. Then run the same warp-uniform guarded copy. Active warps overwrite those zeros before their MMA uses; entirely inactive warps perform no MMA for either M16 fragment, so the known-zero value replaces an irrelevant carried value. Keep pipeline and synchronization placement unchanged. This deliberately trades extra zero-definition instructions for removing conditional carry of old fragment contents; it is not an assumption that fewer loads must be faster.

This is only a proposed compiler-liveness experiment. First verify the exact destinations have no intervening use, compile the isolated variant without executing it, and inspect stack/local instructions against OFF and the failed variant. Reject it before GPU if local traffic remains inflated or zeroing creates substantial new instructions. Even a clean compilation would not establish a speedup: six guarded copies introduce control flow and zeroing overhead, and producer traffic is unchanged. No new repair source or GPU experiment has been created.
