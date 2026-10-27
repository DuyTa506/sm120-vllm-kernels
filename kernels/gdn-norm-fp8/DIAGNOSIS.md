# Exactness failure and isolated follow-up

The original automatic-layout candidate fails exact validation. Preserve
`results/gpu_initial.jsonl`, `results/gpu_initial.stderr`, and
`results/initial_sources/` as the original result/source pair. The rerun in
`results/diagnosis_reproduce.*` reproduces that failure and saves its input
values/reference outputs in `results/failure_01/`.

The trigger is the second changed-input graph replay for T=1025, contiguous
Z, 8 warps, after the T=3 and T=31 cases. It differs by one FP8 byte among
4,198,400 values, with zero scale-bit differences. This is a failure, not an
accepted tolerance. `diagnose_failure.py` is an opt-in GPU probe specifically
for that original kernel. Its diagnostic FP8 bytes equal the original fused
kernel exactly; the installed FLA norm reproduces the saved BF16 reference
exactly. The diagnostic reports eight BF16 norm differences and 5744 rstd
bit differences among 32800 heads.

At token 268, head 16, channel 123:

| Value | Original candidate | Installed FLA/reference |
| --- | --- | --- |
| rstd | 1.035342812538147 | 1.0353426933288574 |
| BF16 norm/gate output | -0.08251953125 | -0.08203125 |
| FP8 output | -22 | -20 |
| Token scale | 0.003923689015209675 | 0.003923689015209675 |

Generated IR identifies different reductions. FLA selects
`ROWS_PER_BLOCK = min(next_power_of_2(ceil(T*32/(2*SM_count))), 4)`.
Offline compilation of the exact installed source confirms:

| FLA rows/block | Values/lane along channels | Channel lanes/head |
| --- | --- | --- |
| 1 | 4 | 32 |
| 2 or 4 | 8 | 16 |

The original 8-warp candidate instead uses 16 values/lane and 8 channel
lanes/head. Each layout accumulates its local values, then reduces across
lanes with descending XOR shuffles. FP32 addition order differs before
BF16 rounding. A CPU float32 replay of the saved failing head gives
119.41014099121094 for the 8-value/lane sum and 119.4101333618164 for the
16-value/lane sum. The variance-plus-epsilon bit patterns differ by one:
1064227343 versus 1064227342. CPU rsqrt estimates match the two observed
GPU rstd values, but this CPU estimate is not GPU rsqrt validation.
`replay_reduction_cpu.py` reproduces this analysis with CUDA hidden.

`fused_explicit.py` is a separate Gluon candidate that subsequently passed
the recorded exact GPU checks (see README and validation summary). It fixes the
per-head layout to the corresponding FLA specialization while allowing
4/8/16 CTA warps. It preserves FLA's `div.full.f32` by 128 and a separate
epsilon addition, the BF16 round trip, and the original compiled-token-FP8
quantization formula. Its wrapper queries the installed rows/block helper;
unsupported values and unsupported warp counts fail. The original kernel
is unchanged. The verifier's `--implementation explicit` is opt-in and
keeps the same exact comparison rules; its default remains `original`.

`compile_explicit_offline.py` compiled FLA rows/block 1/2/4 and six explicit
candidate specializations with CUDA hidden and runtime access forbidden.
Metadata, IR, PTX and assembler logs are in
`results/explicit_layout_offline/`. All six assembled without spills.
Their register/shared-byte counts are 56/16, 40/32, 40/64 for four values
per lane at 4/8/16 warps, and 59/16, 43/32, 38/64 for eight values per lane.
Offline compilation proves neither numerical equivalence nor performance.

The follow-up GPU run included T=3/16/31/1025 and all three warp counts:
144 numerical cases and 72 changed-input graph replays passed. The broader
real-weight run and actual-activation capture also passed; see
[`results/explicit_validation_summary.json`](results/explicit_validation_summary.json).
This resolves the recorded reduction-order failure for the tested inputs
without changing the exact comparison criterion. The original kernel and
failure artifacts remain unchanged. Separate integration paths still need
GPU validation; standalone success does not establish serving behavior.

Minimal repeat command, inside the pinned container with an exclusive GPU:

```sh
python3 /work/kernels/gdn-norm-fp8/verify_bench.py --run-gpu --mode verify \
  --implementation explicit --tokens 3 16 31 1025 --warps 4 8 16
```

Runtime requirements: SM120, the pinned Torch/Triton/vLLM environment,
writable Triton and TorchInductor caches, and the local snapshot for
real-weight/capture steps. Offline scripts use temporary caches, so they do
not prewarm runtime binaries. First GPU execution may compile each layout
and the reference graph; no reliable wall-time estimate has been measured.
Capture additionally loads the full model and writes about 576 MiB of CPU
tensors. Schedule those costs separately from correctness-only microchecks.
