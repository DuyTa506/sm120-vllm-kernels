# Scalar matched-3 Inductor audit

Keep scalar as an **experimental parent** for the isolated `scalar_gdn` versus
`scalar` comparison. The completed pair supports an observed serving improvement,
but neither a scalar-only explanation of that improvement nor quality equivalence.
The four differing `.best_config` files do not by themselves establish four
different runtime kernel choices.

This CPU-only audit reads `exact_matched_3` / `scalar_matched_3` saved cache files.
[The evidence JSON](scalar_inductor_runtime_cpu_audit.json) preserves source and
model-cache hashes, the original tuning records, generated kernel sources, matched
wrapper paths, nested pickle byte offsets/opcodes, compiled configuration hashes,
and ELF section hashes. It executes no pickle reducers, generated Python, Torch,
Triton, GPU work, or HTTP requests.

## Four tuning records mapped to operations

Configurations below are `(XBLOCK, R0_BLOCK if present, warps)`; all use one stage.
Full hashes and generated filenames are in the JSON.

| `.best_config` basename prefix | Generated operation | Saved tuning record: exact → scalar | Linked serialized runtime evidence, each side |
|---|---|---|---|
| `4325999507` | MTP full-attention output gate: flatten BF16 `[T,16,256]`, multiply by sigmoid of BF16 `[T,4096]` | `(1024,4)` → `(512,8)` | One wrapper retains both compiled candidates; final in-memory choice unresolved |
| `519d4a9eab` | MTP embedding lookup + embedding RMSNorm and hidden-state RMSNorm, writing two halves of `[T,4096]` before the BF16 FC projection | `(8,512,16)` → `(2,1024,16)` | One combo wrapper retains six compiled candidates; final in-memory choice unresolved |
| `854a01fdf2` | Native dynamic per-token FP8 quantization of GDN BF16 `[T,32,128]`, flattened to `[T,4096]`, then ordinary CUTLASS out-projection | `(1,1024,8)` → `(1,4096,16)` | Eleven wrappers: ten retain only `(1,1024,8)` on both sides; one retains both candidates |
| `fd2513ff95` | Standalone embedding lookup + RMSNorm; same operation as the embedding half of the MTP combo | `(1,1024,8)` → `(2,1024,16)` | No matching runtime wrapper body found; evidence is a tuning candidate, not an established serving launch |

The generated source file for each of the four entries is byte-identical between
the two caches. AST matching finds 13 runtime wrappers per side, and every matched
wrapper is linked by full source hash to a nested serialized model artifact. The
MTP compile subtree contains the source path `qwen3_5_mtp.py` and attributes
`pre_fc_norm_embedding` / `pre_fc_norm_hidden`. Installed source lines 159–164
perform precisely that embedding / hidden normalization, concatenation, and FC
projection (installed file SHA256
`fbf31b1dd2b63eaf1911e325e3b43db602cf1d4268dfb2d388727e44caffadc2`).
For GDN, the wrapper graph explicitly shows the flatten / abs / amax / scale /
FP8 conversion chain followed by `_C.cutlass_scaled_mm`; its input shape and
4096→2048 projection match the GDN output path.

## What the binaries establish

The ten single-candidate GDN artifacts contain compiled hash
`98b54d8d0549206e653a89b528d4fd5e734aba252363b48a7db98083a5a9ce30`
with `(1,1024,8)` and 32 shared-memory bytes. The first GDN wrapper also retains
`6f06c07b4c88680f605d922054f7827e312b77853c54c04fe33abd2fb048639f`
with `(1,4096,16)` and 64 shared-memory bytes. Both sides contain the same candidate
hash/configuration sets. MTP sigmoid and combo likewise contain identical sets
of two and six compiled candidates, respectively.

For all ten distinct linked compiled candidates, executable ELF sections are
byte-identical between caches. Entire `.cubin` files are **not** byte-identical:
debug sections and their relocations differ; the MTP combo also has differences
in symbol tables and `.nv.global.init`. The JSON records these differences rather
than treating full-binary hashes as equal. This is narrower than proving every
loadable datum or every runtime launch identical.

An initial inspection of literal pickle keys missed memoized `compile_results`
keys and appeared to show a single final MTP choice. The final audit resolves
memoized strings and retains all candidates. **The MTP choices and the first GDN
wrapper's final in-memory choice remain unresolved.** Saved compiler artifacts
are not a per-request launch trace, and this audit is restricted to the four
changed tuning records; it does not claim global determinism.

## Timing and numerical interpretation

The sigmoid kernel is pointwise. Different tiling can affect occupancy and
timing, but there is no floating-point summation whose order changed. The GDN
quantizer uses an absolute maximum, scale floor `1/(448*512)`, reciprocal,
clamping, and FP8 conversion. For finite inputs, max grouping does not have the
sum-order rounding mechanism of RMSNorm. Changing 1024-channel tiles to 4096 and
8 to 16 warps could affect performance if actually selected; this audit does
not establish such a serving change or its size.

The MTP combo does sum FP32 squares over 2048 channels. Changing the reduction
tile from 512 to 1024 changes serial accumulation and reduction grouping, which
can change rounded normalization output and subsequent drafts/acceptance. Its
location in the draft model does not directly explain target-model teacher-forced
prompt NLL. The standalone MTP candidate cannot be assigned a serving impact
without evidence that it launched. None of these possibilities establishes the
cause of the measured timing or quality differences.

In the [serving pair](scalar_vs_exact_matched_3.serving.json), cold prefill median
TTFT is 185.146→164.079 ms at 8192 tokens (11.38% lower), and
384.571→343.632 ms at 16384 tokens (10.65% lower). Those task-level improvements
are larger than the controlled scalar full-call micro effects (about 3% / 2% /
neutral at 4k / 8k / 16k); they must not be relabeled as scalar-only gains.
The parent's 1 Hz telemetry has only 0–6 active samples per cold-prefill scenario,
which also cannot establish a clock explanation.

The [paired repeated NLL report](scalar_vs_exact_matched_3.nll.txt) retains the
`py_email30k` flag: mean 0.794235→0.802070, delta **+0.007835**, with nonoverlapping
three-repeat ranges `[0.790536,0.797625]` and `[0.798475,0.805086]`. Other rows are
mixed. This audit neither dismisses that flag nor proves a model-quality
regression attributable to scalar dispatch.

Proceeding with scalar as an experimental parent is justified by its focused
correctness coverage, all-40-layer eligibility audit, modest micro improvements,
and completed task pair. Adoption remains conditional. The next isolated
`scalar_gdn` versus `scalar` comparison holds the MoE parent fixed and removes
the native GDN norm/quant stage; it can measure that incremental composition,
but cannot retrospectively identify scalar's contribution.

If a scalar-only decision is needed afterward, the minimum useful repeat is
counterbalanced fresh exact/scalar servers under the same FI contract, only the
8k/16k cold-prefill cases and the flagged `py_email30k` cold NLL prompt (same
prompt hash, fresh salts, at least five repeats). Capture final in-memory launch
choices or a focused launch trace if compiler selection is to be tested as a
cause. Another full 17-scenario matrix is not required for this question.

## Reproduction

From the worktree, use the existing dev container solely to read root-owned cache
files; the script imports only Python's standard library:

```bash
sg docker -c 'docker exec -e CUDA_VISIBLE_DEVICES= qwen-prefill-dev python3 -I /work/kernels/moe-nvfp4/audit_inductor_runtime_cpu.py --root /work --output /work/results/qwen_sequential/scalar_inductor_runtime_cpu_audit.json'
```

The recorded audit verifies all eight `.best_config` hashes against the completed
comparison, four equal generated source hashes, 13/13 linked wrapper artifacts
per side, equal compiled candidate configuration sets, and equal executable
sections. It explicitly reports `final_in_memory_selection_fully_resolved=false`.
No serving artifacts, live sources, cache files, or runtime configuration changed.
