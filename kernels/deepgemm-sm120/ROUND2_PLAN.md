# DeepGEMM round 2: representation-compatible FP8 adapter

Status: standalone and fused operator screens completed. Fused QKVZ wins at validated small/medium M on actual captured operands, including strongest-existing-candidate comparisons; large prefill loses. Isolated opt-in vLLM runner is CPU-prepared, serving validation pending. See README for all measurements and scope.

## Audit boundary

The original study pins DeepGEMM `nv_dev` at `b64107f2b9599ca76445b7f62eedf66bae1d095b`, CUTLASS `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`, DeepJIT `e5bdee2bc4ca519eba00cfc5f0c6e950e6a96a16`. The existing image is `dg-sm120-env:1`; the launcher records its immutable ID for each run. Existing ignored compiled source lives under the main checkout, and is mounted read-only. Its existing heuristic diff is recorded without modification. Old results remain immutable.

Old dense measurements compared separately quantized blockscale UE8M0 DeepGEMM inputs against vLLM row/token and channel FP8 inputs. Old grouped FP4 used block32 UE8M0 MXFP4, not the checkpoint's block16 E4M3 NVFP4 plus global scales. These measurements do not reject all native-contract optimizations.

In the pinned source, `csrc/apis/gemm.hpp` permits FP32 output and K scale granularity 32/128 for SM120. `nvfp4_mega_moe.hpp` implements NVFP4 but requires architecture major 10; its presence does not make it an SM120 route. A native SM120 NVFP4 adaptation needs scale-layout and mainloop work, not a dispatch flag. Existing BN128 heuristic experiments are already tested; repeating those is not this round's hypothesis.

## New hypothesis and experiment

Use the exact same FP8 activation and weight bytes as the CUTLASS baseline. Supply packed unit UE8M0 scales to DeepGEMM, accumulate to FP32, then multiply the original arbitrary token/channel FP32 scales in a fused FP32-to-BF16 epilogue. This preserves the quantization representation; differing reduction order can still change results. Unit scales are prepacked, as static metadata would be in an adapter.

Start with Qwen GDN QKVZ (N=12288,K=2048), M=32/64, where the old DG core measurements suggested headroom. Then test M=1024/4096 and GDN output N=2048,K=4096 before considering prefill deployment. These are synthetic shape-matched operands, not checkpoint quality evidence or an integrated vLLM patch.

`bench_rowwise_adapter.py` measures DG core alone and the full adapter separately against same-byte CUTLASS. Both use preallocated BF16 output; DG additionally needs preallocated FP32 scratch. Weight copies exceed L2, with two timing passes in reversed order. Full-output mismatch count and relative L2 are retained, together with a sampled FP64 reference for both implementations. Three changed activation-byte AND scale updates check CUDA graph replay against eager and fresh CUTLASS.

Initial numerical screening: finite output, relative L2 <=1% against CUTLASS, sampled FP64 error no more than 1.1x baseline (floor 1e-6). This is an exploratory falsification gate, not production numerical acceptance. Promote only after stricter representative/checkpoint checks and a practical integrated gain. A core win that vanishes after epilogue is not a deployable win; it can motivate investigating a fused output-scale epilogue.

## Ownership and execution

CPU checks: Python AST parse, no-GPU CLI invocation, `bash -n` launcher passed. GPU must be granted by the root agent and serialized with the shared `gpu.lock`. The launcher refuses result-tag reuse, uses one owned container and isolated writable caches, records source revision/diff and harness hashes, and propagates failures. It does not modify old source/results or install dependencies.

From the qwen-prefill worktree, after slot grant:

```bash
kernels/deepgemm-sm120/run_rowwise_adapter.sh NEW_UNIQUE_TAG --shapes gdn_qkvz --ms 32,64 --iters 30
```

Initial estimate: 3–5 minutes including first JIT compilation. Results and logs go to `results/NEW_UNIQUE_TAG/`. If successful, request a later slot with a new tag for prefill cases. Root owns scheduling and integration; this agent owns these two new harness files and this plan. Combined agentic evaluation remains last, after patches demonstrate individual vLLM task effects.

## Subsequent actual-NVFP4 research

Transfer SM120 tiling/scheduling lessons while keeping per16 E4M3 scales and original global scales. First trace scale-layout consumers in the native NVFP4 backend and evaluate CTA shape/expert work ordering there. Porting the MXFP4 path requires changing its scale grouping, packed representation and associated MMA/layout contract; converting checkpoint scales to UE8M0 is a different quantization experiment and must not be labeled equivalent. Actual router choices must remain unchanged when experimenting with execution order.

## Fused-scale follow-up: operator validated, serving validation pending

`prepare_fused_rowwise.py` copies the actual local source into ignored `fused_src/`, exports `patches/rowwise_epilogue_round2.patch` and fingerprints original source/header/extension bytes. This does not prove equivalence to the intended upstream pin. Patch replay was verified byte-for-byte against the five modified isolated files.

The new `_C.fp8_rowwise_gemm_nt_sm120` binding accepts the existing packed unit blockscale pairs plus original contiguous FP32 row/channel scale arrays. It restricts SM120, contiguous FP8 operands, BF16 output, M>=32 and K divisible by128; host dispatch rejects split-K, swapped operands and accumulation. Coordinate-aware multiplication happens on FP32 accumulators before either TMA/shared or direct output stores, with explicit `__fmul_rn(__fmul_rn(acc,sa),sb)`. Therefore BF16 rounding happens only at final store. Split-K is deliberately unsupported instead of incorrectly scaling partial sums.

CPU host build passed with `python3 setup.py build_ext --inplace` in `dg-sm120-env:1` without GPU access; no packages installed. First build failed due to const-reference mismatch for recipe and is retained as `fused_build.log`; corrected build is `fused_build2.log`. New extension SHA256: `4d994709be878329615b35984a8de88da06669727dbc454873569ef93674b42c`. Subsequent GPU numerical/timing/replay checks passed for the selected QKVZ cells, including actual captured operands; see README. Runtime-selected tested kernels use TMA stores; direct-store variant compiled offline but is not numerically validated on GPU.

Queue after root grants its own slot:

```bash
DG_SOURCE_DIR="$PWD/kernels/deepgemm-sm120/fused_src" kernels/deepgemm-sm120/run_rowwise_adapter.sh rowwise_fused_round2_smoke1 --fused --shapes gdn_qkvz --ms 32,64 --iters 30
```

The same harness retains standalone adapter and core timings, adds fused timing and applies numerical/replay checks to the fused variant. This queue entry has since completed; use fresh tags for future work. Launcher now snapshots four harness files, records actual source fingerprint before/after (including the loaded `.so`), and labels the enclosing repository revision separately. Future promotion additionally requires captured activations/actual checkpoint weight bytes and comparison against the composed existing dispatch; synthetic one-run speedups are insufficient.

Separate prefill boundary queue, original source default, after a dedicated grant:

```bash
kernels/deepgemm-sm120/run_rowwise_adapter.sh rowwise_adapter_round2_prefill1 --shapes gdn_qkvz,gdn_out --ms 1024,4096 --iters 30
```
