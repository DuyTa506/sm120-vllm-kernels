# Actual prefill activation validation

The fixed capture `results/activations_8k_v2` completed successfully with
clean shutdown of its owned engine. The explicit candidate then passed all
9 FP8/scale/GEMM checks and 27 graph replays for layers 0/18/38, plus three
checks reproducing captured BF16 norm outputs. This is instrumentation and
exactness evidence for one request; no latency was measured.

The first capture `results/activations_8k` saved tensors but failed during
worker RPC serialization: `torch.__version__` was a `TorchVersion` subclass.
The worker now serializes `str(torch.__version__)`, and the driver bounds
install/finish RPCs with 60-second timeouts. The initial failure and targeted
cleanup of its verified owned processes are preserved in
`results/activations_8k_runtime_failure.json`. No broad process cleanup or
insecure serialization fallback was used.

`capture_activations.py` creates its own standalone vLLM `LLM`, with eager
execution, compilation mode 0, TP/PP=1, prefix caching disabled, chunked
prefill disabled, no speculation and one sequence. It passes token IDs
directly: exactly 8192 tokens from local Python standard-library source,
encoded with the local snapshot's `tokenizer.json`. No downloads or HTTP
requests occur. Text source and tokenizer hashes are recorded.

Only after the `LLM` constructor finishes, `collective_rpc` invokes the
worker extension's installation method. It finds exactly layers 0, 18 and
38's `linear_attn.norm` modules, verifies the actual norm attributes and
CUDA implementation, then installs pre/post forward hooks. Initialization
profiling and warmup therefore cannot be mistaken for the requested data.
Exactly one norm call of shape `[8192,32,128]` per selected layer is required;
chunking, extra calls or a fused path that bypasses the norm fail capture.

The hooks save X, Z, norm weight and reference BF16 norm output as CPU
contiguous tensors. Separate metadata retains each original shape, dtype,
stride and storage offset. Blocking copies deliberately perturb execution;
no latency is measured. Each layer file is approximately 192 MiB, about
576 MiB total for three layers. File checksums, norm attributes, installed
source hashes, prompt IDs and engine settings accompany the tensors.

The output directory must be new. The driver removes only its own hook
handles and shuts down only its own `llm.llm_engine.engine_core` handle in
`finally`, including on SIGTERM/interrupt. It performs no process-name
kills and never contacts an existing server. Run it as a standalone process
when the experiment scheduler grants the GPU. Do not start it alongside
serving or training.

Inside the pinned container, with this worktree at `/work`:

```sh
# CPU only: preflight imports neither Torch nor vLLM; tests use CPU Torch.
CUDA_VISIBLE_DEVICES= python3 /work/kernels/gdn-norm-fp8/capture_activations.py --preflight
CUDA_VISIBLE_DEVICES= python3 /work/kernels/gdn-norm-fp8/test_capture_io.py

# GPU steps: schedule only after initial synthetic correctness passes.
python3 /work/kernels/gdn-norm-fp8/capture_activations.py --run-gpu \
  --output-dir /work/kernels/gdn-norm-fp8/results/activations_8k_new

python3 /work/kernels/gdn-norm-fp8/verify_bench.py --run-gpu --mode verify --implementation explicit \
  --captured-dir /work/kernels/gdn-norm-fp8/results/activations_8k_v2 \
  --warps 4 8 16 --with-gemm \
  --snapshot /root/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/1c3f884bc99aac2524f6d49bcbac8c88401afd66
```

Without `--run-gpu`, both capture and verifier report pending and return.
The CPU artifact test imports Torch only for CPU tensors and checks CUDA
remains uninitialized. The preflight checks source/API compatibility and
local prompt generation; it does not construct an engine.

`--captured-dir` is verification-only and leaves existing synthetic modes
unchanged. The verifier checks manifest completion, file hashes, shapes,
finite BF16 values and norm settings. It reconstructs the original X/Z/W
layouts, including Z's normal `(12288,128,1)` stride and storage offset
8192, in fresh storage. It checks the installed norm's BF16 output against
the output captured inside the model forward. The fused kernel does not
expose its BF16 intermediate, so that intermediate is not compared directly.

The captured BF16 output then feeds the installed compiled
`QuantFP8.forward_native` reference. Candidate FP8 bytes and FP32 scale bits
must match exactly; installed eager `scaled_fp8_quant` differences are
reported separately. With the command above, snapshot identity and norm
weight bits are checked, and the unchanged checkpoint CUTLASS output GEMM
must also match BF16 bits. Three graph replays verify the captured inputs
without modifying them. Any mismatch stops validation.

These are real activations from an eager execution of the model. They do
not establish identical intermediate activations in compiled production
serving, broad model accuracy, or performance. Successful capture alone is
not kernel validation; a verifier PASS covers only the selected captured
layers and input request. End-to-end serving checks remain necessary.
