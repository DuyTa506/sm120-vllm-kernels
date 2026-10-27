# Offline SM120 compilation

The unchanged `_norm_gate_fp8` prototype compiled and assembled for all three
warp counts with Triton 3.7.1 and Torch 2.13.0+cu130. This establishes offline
code generation only. Numerical agreement, runtime launch, graph replay,
occupancy and performance remain untested.

`summary.json` records source SHA256, specialization, compiler versions,
resource counts and CUDA status. Each `warpsN` prefix contains compiler
metadata, the assembler log, TTIR, TTGIR, LLVM IR and PTX. No device binary
was loaded and no kernel was launched. `CUDA_VISIBLE_DEVICES` was empty;
Torch reported CUDA uninitialized before and after compilation and after
each variant. Guarded Torch CUDA queries and Triton's active runtime driver
recorded no access attempts.

| Warps | Threads/CTA | Registers/thread, ptxas | Shared bytes, metadata | Stack / spill store / spill load bytes |
| --- | --- | --- | --- | --- |
| 4 | 128 | 64 | 256 | 0 / 0 / 0 |
| 8 | 256 | 43 | 32 | 0 / 0 / 0 |
| 16 | 512 | 38 | 64 | 0 / 0 / 0 |

All variants use zero global scratch. Counts describe these assembled
specializations, not runtime occupancy or a preferred warp count. The
explicit `GPUTarget("cuda", 120, 32)` emits `.target sm_120a`.

Compilation used X token stride 4096, Z token stride 12288, epsilon 1e-6,
16-byte pointer alignment and FP fusion enabled. Input/weight pointers are
BF16, output Q is E4M3FN and output S is FP32. This is the actual strided-Z
case; other stride specializations have not been compiled by this audit.

The TTGIR preserves a 32-by-128 logical tensor. Variance reduces axis 1,
then the gated result explicitly truncates FP32 to BF16 and extends it to
FP32. Both the token amax and quantization consume that rounded value.
The amax first reduces the 128 channels, then the 32 heads. Scale-floor,
reciprocal, clamp and FP8 conversion follow the two reductions.

PTX retains `cvt.rn.bf16.f32` followed by `cvt.f32.bf16` before the absolute
maximum and scaling. It uses `cvt.rn.satfinite.e4m3x2.f32` for the final
packed FP8 conversion. These are code-generation observations, not proof
that reduction ordering and rounding match the installed reference.

The 4- and 8-warp layouts use 16 channels per thread, with 4 head groups
and 8 channel lanes per warp. Four warps cover 16 heads per layout tile
and repeat over the 32 heads; eight warps cover all 32 heads in one tile.
The 16-warp layout uses 8 channels per thread, 2 head groups and 16 channel
lanes per warp. Per-head sums and maxima therefore use intra-warp
reductions; the final token maximum communicates across warps. The
generated code includes fused BF16-input/FP32-accumulator operations for
the local sum of squares and a fused multiply/add for variance plus
epsilon. Exact reference comparison must still decide compatibility.

Reproduce inside the pinned container, with `/work` mounted to this worktree:

```sh
sg docker -c 'docker exec -e CUDA_VISIBLE_DEVICES= -e PYTHONDONTWRITEBYTECODE=1 qwen-prefill-dev python3 /work/kernels/gdn-norm-fp8/compile_offline.py'
```

The script extracts only the original JIT function through Python AST,
avoiding vLLM import-time platform selection. It passes the target directly
to `triton.compile`, uses a temporary compiler cache and never accesses
compiled kernel runtime handles. Runtime access raises an error and leaves
a failure report rather than falling back to an active GPU.
