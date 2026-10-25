"""JIT-build the skinny-M FP8 GEMM extension (sm_120a)."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(verbose=False):
    from torch.utils.cpp_extension import load as _load
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0a")
    bd = os.path.join(HERE, ".build")
    os.makedirs(bd, exist_ok=True)
    return _load(name="skinny_fp8", sources=[os.path.join(HERE, "skinny_fp8.cu")], build_directory=bd,
                 extra_cuda_cflags=["-O3", "-std=c++17", "-gencode=arch=compute_120a,code=sm_120a",
                                    "-Xptxas=-v"],
                 verbose=verbose)


if __name__ == "__main__":
    load(verbose=True)
