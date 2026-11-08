# Round 2: actual vLLM PDL and faithful small-M MoE experiments

Status: actual unpinned PDL0/1 diagnostic and the new pinned configuration/NLL
smoke pair are complete. The unpinned pair had retuning confounds; the controlled
pair verifies 18 actual constructor/launch identities and 12 saved-wrapper source
identities with a common reference. Numerical shifts remain unresolved; no pinned
throughput result exists. README and `results/pdl_pin_pair_1.json` are current.
The separate focused driver uses six prefill/four greedy cases and five cold NLL
rounds. GPU execution still needs the root's exclusive slot and shared lock.

## Evidence and untested work

The old synthetic `pdl_inductor.py` experiment produced PDL code: three
PDL-on generated files have gdc_wait and launch_pdl, none off do. Nevertheless
both graph medians are125.1us, with zero consecutive overlaps. That chain
alternates norm/quant code and non-PDL BF16 cuBLAS calls. It does not test the
actual vLLM graph's existing PDL-aware FlashInfer consumers. Its eager-error
print is also not a strict on/off output equivalence check.

Installed Torch2.13 source in the vLLM0.32 image confirms
`config.triton.enable_pdl = TORCHINDUCTOR_ENABLE_PDL == '1'` (defaultfalse).
`codegen/triton.py:_enable_pdl_codegen` emits GDC operations on supported CUDA
but explicitly excludes TritonTemplateKernel. Therefore an environment toggle
is a supported candidate, not proof that every desired graph edge participates.

The old small-M MoE timings are also incomplete for present integration: M4
uses a real fourth token rather than a masked graph tail, assumes unit activation
scales rather than the checkpoint's calibrated scales, and samples routes from
prefill. The new route captures show three real target rows at c1 and unresolved
instrumented output equivalence. Do not reuse the old claim that c1 necessarily
routes an extra padded fourth row.

## Experiment A — PDL in the actual serving graph

Hypothesis: enabling dependency-aware Inductor producers can expose useful
prefetch/launch overlap with existing FlashInfer consumers, unlike the old
synthetic cuBLAS-bound chain. Test on the scalar stack first; the only deliberate
change is `TORCHINDUCTOR_ENABLE_PDL=0/1`.

1. Root runner needs an explicit validated PDL setting passed into the container,
   recorded in config/source provenance and allowed by runtime comparison.
   Do not silently smuggle it through a reused cache. Use separate fresh
   VLLM_CACHE_ROOT, TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR for0 and1.
2. Before inference timing, inspect generated actual backbone/MTP code for
   gdc_wait/gdc_launch_dependents/launch_pdl; record producer and consumer
   coverage. A launch attribute without a cooperative edge is not a success.
3. Check matched compiled producer outputs and changed-input CUDA graph replays
   with actual shapes/strides and fixed inputs. Compare PDL0 against PDL1, not
   just each compiled version against an eager reference. Preserve BF16/FP8
   bytes/scales wherever the reference defines exactness.
4. Short serving diagnostic: existing greedy c1/4/16/64 plus existing cold
   prefill sweep, same132-entry FI reference through shutdown and same source
   snapshots. Repeat reversed order if apparent improvement is small. Capture
   short nsys graph traces separately from timing to identify changed overlap.
5. If worthwhile, run the existing17-case suite and quality diagnostics. Agentic
   remains only the final newly selected composition, never individual PDL.

Original launch requirements (now implemented in directory-local runners): `FI_CONTROL=1`, variant scalar, suite focused,
`TORCHINDUCTOR_ENABLE_PDL=0|1` passed **inside** the server container, distinct
cache roots per setting, GPU budget0.78, MTP2. Existing qwen_experiment.sh does
not forward this host variable or make its cache root unique by PDL mode; merely
prefixing the current command with that environment variable would not test it.

## Experiment B — cooperative small chain if A has insufficient coverage

Use actual captured norm/quant inputs and one existing PDL-aware FlashInfer
consumer. Compare four modes: neither, producer only, consumer only, both.
Retain the coherent-load fix from pdl_chain.cu; the archived v1 noncoherent-load
hoisting bug makes it unsuitable. Include cold weights exceeding L2 and changing
activations on every replay. Count GPU launches/overlaps separately from clean
CUDA-event timing. Only then consider adding PDL hooks to skinny BF16/FP8 GEMMs;
those files belong to other tracks and must not be changed concurrently.

## Experiment C — small-M NVFP4 dispatch, distinct from swap-AB

Do not count already-existing CUTLASS swap-AB as new work. Compare M1/2/3/4/6/8
with the real valid-row masks and checkpoint activation-scale contract. Separate
B12x direct_micro (<=3) from MMA micro paths; their scale conventions differ.
A mode-specific reciprocal adapter is a hypothesis to validate against an FP32
reference and the calibrated CUTLASS parent, not permission to replace scales
with1. Nonuniform, extreme and boundary scales must be included. If the backend
cannot represent the checkpoint contract, reject that dispatch rather than
claiming its unit-scale timings transfer to the model.

The first deliverable should be numerical/full-apply micro evidence on actual
layer weights and target/draft-shaped routes, with changing-route graph replay.
Only a passing candidate gets guarded small-M vLLM dispatch and NLL/MTP checks.

## New independent hypothesis

The old trace contains41 memset nodes around MoE output initialization. Test
whether output zeroing can be combined with an existing activation/setup launch
while preserving all untouched/padded output rows and cross-stream ordering.
This is a FlashInfer implementation experiment, not a PDL attribute toggle.
Measure full-call savings; do not infer35us saved just by adding node durations.
Root should coordinate this with the MoE glue-fusion track to avoid duplicate work.

## GPU ownership

Use the established lock:
`/tmp/claude-1000/-home-aiqk3-Desktop-duy-dev-training-space-gpu-unlock/2207ad44-a66d-418b-840d-33bbf11f51c5/scratchpad/gpu.lock`.
Require no other compute process, a fresh result tag and uniquely owned disposable
container; verify removal and source hashes before releasing the slot. The old
run scripts overwrite named output files and therefore should not be executed
unchanged as new evidence. No new benchmark speedup is claimed in this plan.

## Implemented round-2 protocol

`run_vllm_pdl.sh TAG 0|1` is the directory-local frozen launcher. It explicitly
forwards PDL and three fresh cache roots, snapshots itself and its provenance
helper in addition to the common runtime manifest, and uses fixed FI tactics.
The first smoke intentionally runs greedy c1/4/16/64 and cold NLL only; the
focused prefill sweep is the next pair, not silently added midway through this
pair. No agentic work is run here.

`audit_pdl_artifacts.py` matches exact wrapper source hashes inside safe pickle
opcode inspection of the backbone/eagle AOT models named by this startup log.
PDL0 binds 14 backbone and 2 eagle wrappers with zero launch_pdl markers. This
is current saved-model evidence, not per-request tracing or proof of overlap.
Its CPU test rejects a modified wrapper and excludes unrelated cache files.

`compare_pdl.py` creates disposable normalized artifact copies, validates the
original PDL/cache/source contract first, and delegates to existing strict greedy
and runtime checks. Allowed rewrites are reported: container-name prefix, result
mount path, explicit PDL mode and isolated cache roots. Originals are untouched.
The adapter still requires real paired artifacts for full validation.

## Concrete next boundary and prefill protocol

PDL1 current-model linkage confirms all 14 backbone and 2 eagle saved wrappers
contain `launch_pdl` and `gdc_wait`; PDL0 contains neither. Fresh PDL1 PTX includes
`griddepcontrol.wait` and `griddepcontrol.launch_dependents`. Participation is
therefore viable regardless of the first decode timing result. Run a **new**
focused pair, fresh caches and frozen revised launcher, using existing
`serving_matrix.py --suite prefill` (six original prompt lengths), then identical
greedy/NLL probes if requested. Do not append prefill midway through this first
pair or reuse its timing as the new baseline. No per-patch agentic benchmark.

For a cooperative-boundary experiment, select an exact saved wrapper's
`triton_red_fused_*cutlass_scaled_mm*` producer and immediately following
`torch.ops._C.cutlass_scaled_mm.default` consumer. The Python consumer wrapper
ultimately dispatches the native `_C` op; there is no PDL flag exposed in its
Python signature. That observation does not prove the native kernel lacks PDL.

CPU audit first: hash the installed `_C` binary and match the selected consumer
kernel symbol with a short node trace; inspect its PTX/SASS for dependency wait
instructions and inspect the native launch path for programmatic serialization
attributes. Obtain the exact matching native source/build identity before any
patch. For CUDA graphs inspect the relevant node attribute, not merely eager
launch flags. Record the actual kernel symbol, argument shapes/strides/scales,
stream and predecessor/successor node IDs. Trace is separate from timing.

If the consumer is cooperative, measure its existing edge with PDL producer
on/off. If it is not, build an isolated consumer experiment with an explicit
wait before reading producer-owned activation/scales, retaining weight prefetch
before wait only when valid. Compare neither / producer only / consumer only /
both. Validate exact FP8 activation bytes and scales plus BF16 output against
the original op on changing-input graph replays, multiple actual M shapes, and
weights exceeding L2. Compare full edge timing, not sums of unrelated kernel
medians. Graph launch metadata, native wait and producer signal are three
separate gates; speed delta alone proves none of them.

## Controlled baseline constructor recorder

Historical AOT extraction now reaches 96 serialized CachingAutotuner records,
but its selection reconstruction intentionally rejects 17 unresolved cases.
Rather than silently choosing historical winners, the next protocol records
**all actual constructor candidate sets** with exact body/signature/size and
compiler alignment/options metadata. `canonicalize_pin_record.py` chooses the
lexicographically first configuration in the intersection across each identity's
constructors. Empty intersections, identity collisions, unknown kernels and
missing/duplicate configs fail closed. This defines a NEW controlled baseline.

`run_vllm_pdl_pin.sh TAG 0|1 record|pin [REFERENCE]` starts an isolated server
and collects ten cold NLL requests; it has no greedy throughput workload.
`pin_bootstrap/sitecustomize.py` installs a hash-guarded Torch hook in all Python
subprocesses. Bootstrap errors terminate the process with exit 86, because
Python ordinarily swallows sitecustomize exceptions. Torch subprocess source
inspection confirms its environment inherits the current sys.path, including
this explicit bootstrap path. CUDA-hidden CPU bootstrap succeeds; a bad
reference hash terminates before user code. Full pin sources are snapshotted.

The hook pins before compilation, including warm-cache-only workers, disables
postcompile dynamic R-block and coordinate-descent tuning, rejects combo tuning,
and asserts singleton compiled/launcher/final launch configurations. No global
runtime file is modified. Final paired coverage must match identities and final
configs; recording alone does not establish complete pin coverage or correctness.

## Preregistered numerical investigation budget

`pdl_numerical_policy.json` requires five cold rounds per variant and flags
absolute prompt-mean NLL shifts or within-mode ranges above 0.001 nat/token.
The threshold is a conservative scheduling-change investigation budget (about
0.1% perplexity multiplier), not a model-quality gate. The smoke's logprobs-2
delta −0.146537 is unresolved despite identical prompt/token hashes.

If within-mode repeats are stable but cross-mode shifts remain, localize with
fixed actual hidden-state/norm/quant/GEMM operands: compare producer BF16 values,
FP8 bytes and row scales separately, then native GEMM outputs. Repeat unchanged
inputs to distinguish native reduction nondeterminism; replay changed inputs to
catch stale data/dependency races. Existing DeepGEMM GDN captures contain already
quantized projection operands, so they cannot reconstruct upstream norm input
and are insufficient to attribute producer drift without an additional capture.
