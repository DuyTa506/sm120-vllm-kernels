import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["FUSED_ROUTER_VERBOSE"] = "1"
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
import fused_router as fr
t = time.time(); fr._ext(); print("built in", time.time() - t, "s")
