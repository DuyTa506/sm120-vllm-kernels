# Corrected FI MMA bucket comparison

This independently verifies the scale contract already recorded in `../moe-nvfp4/README.md` §4c and `results/small_m_L19.txt`; it is a correction to the comparator, not a new kernel optimization. MMA micro/static use divisor scales with `input_scales_are_reciprocal=False`; direct uses multiplier scales. The previous M8 auto comparator used the wrong setting and produced zero output.

The initial isolated M8 numerical probe confirms both packed input and SFA bytes exactly match vLLM quantization at captured offsets0/64/restore0 with the corrected setting. All old-flag outputs remain zero. See `CPU_AUDIT.md` for derivation and source fingerprints.

## Paired bucket screen

Real L19 weights and captured activations, full-call CUDA graphs, two timing orders. Both FI providers include identical safe-input sanitation; M4 MMA includes route compaction. Current scalar CUTLASS includes input quantization. Router, weight preparation and initial buffer allocation are excluded. “Padded” means the last token has all eight routes invalid, so M4 has three live tokens and M8 has seven.

| Bucket | Corrected MMA (µs) | Sanitized direct (µs) | Current CUTLASS (µs) |
|---|---:|---:|---:|
| M4, four live | 28.701 / 28.726 | **26.670 / 26.680** | 61.225 / 61.139 |
| M4, three live | 32.836 / 32.835 | **26.705 / 26.672** | 49.948 / 49.990 |
| M8, eight live | **34.805 / 34.834** | 43.024 / 43.027 | 63.486 / 63.459 |
| M8, seven live | **34.802 / 34.788** | 43.066 / 43.059 | 61.331 / 61.325 |

M4 uses forced native micro; M8 uses forced static. These are repeated hot-layer timings, not vLLM request speedups. The best tested FI backend depends on bucket size. The production global hybrid threshold remains unchanged.

Numerical checks precede timing: captured offsets0/64, last-token padding, all-invalid, restored input. All pass the exploratory FP32 relative-L2 gate of 0.25; observed full-input errors are about0.134–0.147, comparable to CUTLASS quantization error. Corrected M4 micro has graph/eager/repeat variation up to0.009196 relative L2; direct and CUTLASS were exact in this run. All M8 providers were exact on repeat/eager checks. Padded/all-invalid output rows are exactly zero. These gates do not replace model-quality evaluation, and no serving promotion is claimed.

Both jobs exit0; owned children and process groups are absent; post-run GPU queries are empty. Harness manifests, all28 runtime sources, capture hash, checkpoint identity records and image are equal across the M4/M8 runs. Checkpoint identity includes index and input-scale tensor hashes plus shard file identity; it is not a full content hash of every weight shard.

Artifacts:

- `results/fi_static_scale_l19_m8_1/measurement.json`: original wrong-flag versus corrected numerical experiment.
- `results/fi_mma_bucket_l19_m4_1/measurement.json`
- `results/fi_mma_bucket_l19_m8_1/measurement.json`
- `results/paired_bucket_summary.json`: compact metrics, raw samples, numerical limits, cleanup and cross-run parity.

Each result directory preserves source snapshots, runtime source copies/hashes, checkpoint/capture identity, process lifecycle and terminal verification. `vendor_direct/origin.json` identifies the exact existing sanitizer/native-direct helpers copied for this isolated test; their source directory was not modified.
