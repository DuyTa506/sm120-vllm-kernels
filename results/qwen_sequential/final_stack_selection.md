# Selected stack for the final Qwen agentic comparison

Select **`scalar`**, consisting of root `next` patches 01–05 plus MoE patches
03–06, for the final comparison against `next`. Keep the additional GDN
norm/FP8 overlay opt-in and unselected for this measured configuration. The
choice prioritizes the scalar composition's observed prefill gains while
avoiding the repeated incremental greedy c=4/c=64 throughput losses from GDN.
It does not establish broad model-quality equivalence, promote this stack to
default `next`, or reject GDN fusion for other workloads.

Target: `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast`, vLLM 0.32.0, FlashInfer
0.7.0.post1, RTX PRO 6000 Blackwell SM120, TP1/no EP and MTP2. The selected
MoE path remains hybrid: b12x at >=1024 tokens and the exact-alpha CUTLASS
fallback below that threshold. The model's learned expert routing is unchanged.

## Exact composition

Patch numbers are directory-local. Apply root 01–05 in their existing order,
then MoE 03→04→05→06 to the matching pristine vLLM MoE/oracle sources.
The measured runner uses the already generated `build/` files and frozen
`scalar/` overlay, mounting the patched MoE oracle alongside it.

| order | patch | retained behavior |
|---|---|---|
| root 01 | [SM120 FP8 prefill and page64](../../patches/vllm-v0.32.0/01-sm120-fp8-prefill-and-page64.patch) | original FP8 attention integration prerequisite |
| root 02 | [GDN output/checkpoint](../../patches/vllm-v0.32.0/02-gdn-inplace-output-and-internal-checkpoint.patch) | in-place output and internal prefill checkpoint |
| root 03 | [FP8 prefill without glue](../../patches/vllm-v0.32.0/03-sm120-fp8-prefill-noglue.patch) | existing fused attention prefill path |
| root 04 | [kernel page override](../../patches/vllm-v0.32.0/04-kernel-page-override.patch) | attention page16 with the existing 2176-token cache block |
| root 05 | [GDN convolution fusion](../../patches/vllm-v0.32.0/05-gdn-fused-conv-post-conv.patch) | existing fused convolution/post-convolution path |
| MoE 03 | [scale fix and hybrid](../../kernels/moe-nvfp4/03-moe-b12x-scale-fix-and-hybrid.patch) | corrected b12x scale convention, guarded hybrid and MTP oracle fallback |
| MoE 04 | [shared workspace](../../kernels/moe-nvfp4/04-moe-b12x-shared-workspace.patch) | reuse scratch across compatible layers |
| MoE 05 | [exact CUTLASS alphas](../../kernels/moe-nvfp4/05-moe-cutlass-exact-alphas.patch) | retain alpha arithmetic from original processed scales |
| MoE 06 | [scalar FC1 input scale](../../kernels/moe-nvfp4/06-moe-b12x-scalar-input-scale.patch) | load-time uniform-scale representation selects the existing shared-input dispatch |

The runner preserves `CONFIG=next` and adds
`VLLM_B12X_FIX_SCALES=1`, `VLLM_B12X_SHARED_WORKSPACE=1`,
`VLLM_B12X_SCALAR_INPUT_SCALE=1` and `--moe-backend flashinfer_b12x`.
The scalar hook requires the recorded Qwen geometry and exactly uniform,
positive, finite FC1 scales. Unsupported scales retain the vector path;
scalar selection does not change FC2 or weight-alpha values.

The additional [GDN norm/FP8 patch](../../kernels/gdn-norm-fp8/01-gdn-norm-fp8-experimental.patch),
its five-file overlay and its extra compilation split are **not part of this
selected stack**. Existing root GDN patches 02 and 05 remain included.
The [CPU patch verifier](../../kernels/gdn-norm-fp8/verify_patch_composition_cpu.py)
reproduces every MoE intermediate and the final scalar overlay; it also proves
that the optional GDN and scalar changes apply in either order after their
prerequisites. Packaging compatibility does not require adopting both.

## Measurement basis

The completed controlled `next_matched_3` / `scalar_matched_3` pair covers all
17 serving tasks, four greedy cases, 30 cold NLL calls and 20/20 needles per
configuration. The [serving report](scalar_vs_next_matched_3.serving.md) has
cold 8k/16k/32k TTFT changes of -18.8%/-9.4%/-7.1%, seven workload throughput
changes of -1.4% to +5.2%, and [greedy rates](scalar_vs_next_matched_3.greedy.md)
of -3.70% to +3.67%. These describe whole measured configurations, not isolated
scalar-dispatch causation or a timing confidence interval.

The incremental GDN evidence includes both a [full pair](gdn_vs_scalar_matched_3.serving.md)
and a [focused pair](gdn_vs_scalar_focus_1.prefill.md) run in reversed order,
GDN then scalar. Both focused processes exit 0, their captured 29/23 sources
match the manifests, all 30 GDN adapters are ready, and each completes six
prefill cases, four greedy cases and 50 cold NLL calls. Both configurations
have the same shorter prior suite; it differs from the original full suite.

| incremental GDN effect | original full pair | reversed focused pair |
|---|---:|---:|
| 8k cold-prefill TTFT | +7.40% | -7.31% |
| 16k cold-prefill TTFT | +1.40% | -0.75% |
| 32k cold-prefill TTFT | -7.57% | -1.98% |
| greedy c=1 output rate | +2.17% | +1.58% |
| greedy c=4 output rate | -3.74% | -4.18% |
| greedy c=16 output rate | +5.00% | -7.70% |
| greedy c=64 output rate | -8.71% | -4.04% |

The 8k effect changes direction and the 32k gain shrinks, while c=4/c=64
losses recur. This supports the current scalar selection. The [complete GDN
record](../../kernels/gdn-norm-fp8/results/COMPOSED_SERVING.md) retains its useful
microbenchmarks, 32k/long-doc improvements and scoped state proof for further
work; no general claim that GDN fusion is slower follows from this selection.

## Controls and limitations retained

The full and focused pairs use GPU budget 0.78 and identical fixed-reference
FlashInfer tables at startup and shutdown: 132 entries, effective-entry SHA256
`721ed789107ca7cd8d54b4ce8e6345a9e84e8882beca5914a4a2085598f1b063`.
Inductor is observed rather than pinned. Saved tuning records can differ without
proving final per-launch choices; compiler and scheduling causation remain open.

At the same GPU budget, logged KV availability/final graph pool is
44.61/4.90 GiB for `next`, 45.50/3.45 GiB for scalar, and 45.62/3.33 GiB for
scalar+GDN. These are startup context, not equal KV capacities or peak-memory
measurements. Model sampling, MTP acceptance and generated paths can vary;
even explicit temperature zero does not make the whole model bitwise equal.

The earlier scalar-versus-exact `py_email-30000` NLL flag remains recorded
(mean 0.794235→0.802070 with nonoverlapping three-repeat ranges). Against the
fresh `next`, scalar's same mean delta is +0.002717 with overlapping ranges.
The original GDN stack versus `next` has a `licenses-30000` flag of +0.010715
with nonoverlapping three-repeat ranges. In the [focused GDN/scalar NLL pair](gdn_vs_scalar_focus_1.nll.txt),
eight of ten means rise, all five-repeat ranges overlap, and the licenses delta
is only +0.000125. That shorter pair does not include `next`, so it does not
retest or erase the earlier stack-versus-next result. None of these small
corpora, overlapping ranges, or 20/20 needles establish quality equivalence.

## Final agentic execution

Only the selected complete stack enters the final agentic pair. The recorded
run tags are `next_final_agentic_1` and `stack_final_agentic_1`; their results
are pending completion and are not asserted here. Existing output tags are
protected: use fresh tags for a later reproduction, with an otherwise idle GPU.

```bash
sg docker -c 'set -e
FI_CONTROL=1 GREEDY_DECODE=0 NLL_ROUNDS=0 NOSPEC=0 bash bench/qwen_experiment.sh next_final_agentic_1 next agentic
FI_CONTROL=1 GREEDY_DECODE=0 NLL_ROUNDS=0 NOSPEC=0 bash bench/qwen_experiment.sh stack_final_agentic_1 scalar agentic
'
```

The [agentic protocol](../../bench/AGENTIC_COMPARISON.md) captures both production
input sets before replay, verifies them after replay, checks all eight workload
shapes and preserves the separate live-short functional smoke. The common
runtime sources, fixed-FI gates and source verification remain required through
shutdown. No agentic replay is run for individual patches. Replay measures fixed
prompt/output-length performance, not live agent-quality equivalence; review
the live-short smoke separately and retain any failure.

For the completed focused comparison, the recorded invocation was:

```bash
sg docker -c 'set -e
FI_CONTROL=1 NLL_ROUNDS=5 bash bench/qwen_experiment.sh gdn_focus_1 scalar_gdn focused
FI_CONTROL=1 NLL_ROUNDS=5 bash bench/qwen_experiment.sh scalar_focus_1 scalar focused
'
```

The source-faithful patch check is CPU-only and writes a fresh report:

```bash
python3 -I kernels/gdn-norm-fp8/verify_patch_composition_cpu.py \
  --output /tmp/qwen_patch_composition_recheck.json
```

Root prerequisites are generated by [scripts/build_patched.sh](../../scripts/build_patched.sh)
from the pinned vLLM image; do not rebuild runtime sources during an active
measurement. The final runner mounts the frozen overlays and records their
hashes rather than modifying the installed package.
