# Targeted packed GDN state probe: initialization failure retained

The first owned GPU attempt (`packed_state_probe_1`) exited1 during MTP model
construction, before hook installation or comparisons. The fixture forced
`flashinfer_b12x`, but its narrow mount set did not include the serving stack's
unquantized-MTP oracle compatibility patch. Installed vLLM rejects that backend
for the ignored/unquantized MTP experts. The owned container was removed and
the GPU compute list was empty when the slot was released. No retry ran.
See `results/packed_state_probe_1_failure.json` and the original log/manifest.
The probe2 fixture now explicitly uses auto MoE and CPU preflight rejects a
forced backend with these four mounts. Its GPU attempt requires separate scheduling.
Failure1 sources are retained in `results/packed_state_probe_1_sources/`. This failure provides no GDN
numerical or state-parity result.

This separate probe compares the overlay's `packed_norm_fp8` against the
unchanged baseline `_forward_core_fused_norm_packed` plus **explicitly compiled
raw `QuantFP8.forward_native`**. Eager model quantizer dispatch itself is not
the oracle. No serving source, overlay, launcher or frozen verifier is changed.

`probe_packed_state.py` creates one owned eager TP1/MTP2 LLM and installs a
worker extension after initialization. Two local standard-library token
prompts (64 and8192 tokens), a2048-token chunk budget, two sequence slots and
32 output tokens encourage actual mixed scheduling. Only layer18 is observed.
The worker tests at most five cases: first long prefill, first observed mixed
spec/non-spec cases, first pure decode, and a checkpoint-bearing case if one
has not already been covered. Completion requires observed long prefill, pure decode, and a mixed
prefill+decode case with at least1024 actual tokens whose recorded route is
`fused` at both original T and T+17. Tiny mixed cases remain observed but do
not consume the required mixed-case slot. Missing coverage fails explicitly;
checkpoint and mixed-spec/non-spec coverage are reported individually.

Before each selected live original call, it makes two private shallow owner
copies sharing only read-only weights/submodules. Each has private compact
convolution/recurrent caches initialized from the actual referenced rows.
All metadata tensors, nested aliases and checkpoint scratch are cloned;
state indices (including cached checkpoint-export indices) are remapped.
The installed GDN null-row convention must be0;0 and pad slot−1 are preserved.
Original method descriptors must bind the private owner. A temporary copied
forward context supplies only the private remapped metadata and is restored
by vLLM's existing context manager.

For original T and T+17, the two routes receive independent identical packed
inputs and zero core buffers. Added QKVZ/BA rows contain NaNs. Checks require:

- Exact FP8 bytes and scale bytes, plus every byte of both private updated
  caches, including unchanged/null rows; NaNs are compared by bits.
- Padded Q bytes are zero and scales equal `1/(448*512)`.
- Adding padding leaves actual outputs and updated private caches unchanged.
- Full selected-layer live cache snapshots are byte-identical before/after
  private comparisons, including rows outside the private compact selection.

The live cache is never replaced or restored. After successful checks, the
saved original bound method runs once on the original owner to advance its
real trajectory. The probe refuses CUDA stream capture or a non-eager forward
context. It does not double-run a stateful CUDA graph. `cases.jsonl` is flushed
per attempted case, including live-cache mismatch counts on failure. Failure
stops the owned request; no candidate result enters the live model.

The compact private caches are unrelated to live scheduler allocation. This
checks the real core's semantics on observed metadata/state/inputs and tests
padding at that boundary. It does not prove allocator identity, asynchronous
scheduler behavior, CUDA graph replay, all accepted-token patterns, every GDN
layer, paired generation/NLL equivalence, or performance. Existing compiled
integration smoke covers startup/capture/execution; standalone checks cover
norm/quant/GEMM and changed-input graph replay. These are complementary scopes.
Checkpoint rows and accepted-token values appear in the report only when
observed; their absence is not converted into a pass for that coverage.

## Commands, ownership and prerequisites

The host runner uses the existing GPU lock nonblockingly, checks for an active
GPU compute process, and starts a fresh randomly named container. It mounts
the same four `next` build files read-only and verifies the baseline GDN SHA
`c82a390240c77032414d916ed705780716cd696f90722eb99eb89419f0aa262c`.
The isolated candidate helper is imported under its own module name; it never
replaces the installed GDN module. Fusion remains disabled in the live model.
The corrected fixture obtains activations from the baseline auto MoE backend.
The integrated B12x serving smoke is separate evidence; this is not a matched
serving-stack quality comparison.

From the worktree root, CPU commands:

```sh
python3 kernels/gdn-norm-fp8/probe_packed_state.py
python3 kernels/gdn-norm-fp8/run_packed_state_probe.py  # prints plan, launches nothing
sg docker -c 'docker exec -e CUDA_VISIBLE_DEVICES= qwen-prefill-dev python3 /work/kernels/gdn-norm-fp8/test_packed_state_probe.py'
sg docker -c 'python3 kernels/gdn-norm-fp8/run_packed_state_probe.py --preflight --output-dir kernels/gdn-norm-fp8/results/packed_state_preflight_1'
```

The last command owns a CPU-only fresh container with `CUDA_VISIBLE_DEVICES=`
and `NVIDIA_VISIBLE_DEVICES=void`; no GPU device request is passed. The saved
preflight run in the existing development container intentionally reports
`runtime_source_ready=false`: that container's installed source differs from
the required baseline build. GPU mode rejects it; the fresh runner supplies
the required mounts. CPU preflight imports neither Torch nor vLLM. A fresh CPU-only mounted preflight
also completed exit0 with `runtime_source_ready=true`; its exact command,
ownership label and successful result are in
`results/packed_state_preflight_1.runner.json` and
`results/packed_state_probe_mounted_preflight.log`. No GPU was requested.

**Only after root grants the exclusive GPU slot:**

```sh
sg docker -c 'python3 kernels/gdn-norm-fp8/run_packed_state_probe.py --run-gpu --output-dir kernels/gdn-norm-fp8/results/packed_state_probe_2' \
  > kernels/gdn-norm-fp8/results/packed_state_probe_2.log 2>&1
```

The local snapshot must exist under `/opt/intramind/hf-cache` (override
`--hf-cache` if needed), and the two cache directories must be writable.
Model initialization and first native-quant/kernel compilation dominate this
small probe; allow minutes, depending on cache state. It records no timing.
Existing output directories/runner records are rejected. Cleanup uses the
owned LLM handle and only a container whose random name **and ownership label**
match this invocation. Stop targets the verified immutable container ID; its exit status is recorded
and container state is re-inspected. If still alive, the label/name are rechecked and force removal
targets that immutable owned container ID. Cleanup is accepted only when the
container is absent or terminal (exited/dead, not running/restarting, PID0).
A created or inaccessible container is not a cleanup pass. Unverified GPU
cleanup records failure and retains the GPU lock while polling only that
container; no slot release is reported while it may be alive. Manual recovery
is recorded and still leaves the run failed. There are no process-name kills
or HTTP calls.

Twenty CPU isolation/route/runner/backend/cleanup regressions pass, including
stop failure/timeout, ownership change, daemon failure, force-removal failure,
in-flight created containers, and a real temporary-file flock test proving
the lock stays held until cleanup recovery is verified. The mounted probe2
preflight exits0 with runtime_source_ready=true and imports neither Torch nor
vLLM; `results/packed_state_probe_2_guarded_preparation.json` records current hashes
and the fresh mounted preflight with verified container removal. They cover null/negative
index remapping, optional index fields, nested checkpoint/scratch copies,
proxy method binding, original-hook preservation, live-cache independence,
bitwise NaN comparisons, route labels, and GPU-free preflight construction.
They are preparation evidence; GPU state parity remains pending.
