import sys; sys.path.insert(0,'/work')
import glob
print(glob.glob("/hf/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/*"))
from safetensors import safe_open
import vllm
from vllm import _custom_ops as ops
print(hasattr(__import__('torch').ops._C,'fused_gdn_decode_post_conv_mtp'))
import common
common.build()
print("ok")
