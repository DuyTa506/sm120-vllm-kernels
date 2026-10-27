"""Allocation/lifetime regression tests, using actual FlashInfer GPU buffers.

Run in the vLLM image with --gpus device=0:
    python3 -m pytest test_shared_workspace.py -q -s
The backend is loaded from shared/ (or B12X_TEST_BACKEND for the baseline).
No model weights or expensive GEMM compilation are needed here.
"""

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.v1.worker import workspace as ws


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("VLLM_B12X_FIX_SCALES", "1")
    monkeypatch.setenv("VLLM_B12X_SHARED_WORKSPACE", "1")
    path = Path(os.environ.get("B12X_TEST_BACKEND", str(
        Path(__file__).parent / "shared/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py"
    )))
    spec = importlib.util.spec_from_file_location("b12x_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ws.init_workspace_manager(torch.device("cuda:0"), num_ubatches=2, num_lanes=2)
    yield module
    ws.reset_workspace_manager()
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


def make_expert(backend, max_tokens=16384, device="cuda"):
    # Bypass weight loading only. _ensure_wrapper and every underlying GPU
    # allocation are real; the bug does not depend on checkpoint contents.
    expert = object.__new__(backend.FlashInferB12xExperts)
    expert.global_num_experts = 256
    expert.num_local_experts = 256
    expert.topk = 8
    expert.hidden_dim = 2048
    expert.intermediate_size_per_partition = 512
    expert.max_num_tokens = max_tokens
    expert._activation_str = "silu"
    expert.out_dtype = torch.bfloat16
    expert._wrapper = None
    scale = torch.ones(256, device=device)
    expert.quant_config = SimpleNamespace(
        w1_scale=scale, w2_scale=scale, g1_alphas=scale, g2_alphas=scale,
    )
    expert._fc2_input_scale = scale
    expert.w1_sf_mma = scale
    expert.w2_sf_mma = scale
    expert._hybrid_min_tokens = 1024
    return expert


def wrapper(expert):
    result = expert._ensure_wrapper()
    # Baseline returns None and stores a single wrapper on the layer.
    return expert._wrapper if result is None else result


def test_identical_layers_do_not_duplicate_large_gpu_buffers(backend):
    a, b = make_expert(backend), make_expert(backend)
    wa = wrapper(a)
    before = torch.cuda.memory_allocated()
    wb = wrapper(b)
    extra = torch.cuda.memory_allocated() - before
    print(f"second_layer_extra_mib={extra / 2**20:.3f}")
    assert extra < 16 * 2**20, "scratch allocation still grows per layer"
    assert wa is not wb, "weight-view caches must remain layer-specific"
    # The allocated scratch/output memory must be reused while final outputs
    # are copied out by apply(). Test actual storage reuse, not mock calls.
    wa._moe_output.fill_(7)
    assert torch.all(wb._moe_output == 7)


def test_different_capacities_do_not_share_output(backend):
    a = wrapper(make_expert(backend, 1024))
    b = wrapper(make_expert(backend, 2048))
    a._moe_output.fill_(3)
    b._moe_output.fill_(5)
    assert torch.all(a._moe_output == 3)
    assert torch.all(b._moe_output == 5)


def test_same_layer_resolves_current_lane(backend):
    expert = make_expert(backend, 1024)
    first = wrapper(expert)
    first._moe_output.fill_(3)
    with ws.use_workspace_lane(1):
        other = wrapper(expert)
        other._moe_output.fill_(5)
    assert torch.all(first._moe_output == 3), "lane 1 overwrote lane 0"
    assert wrapper(expert) is first


def test_same_layer_resolves_current_ubatch(backend):
    expert = make_expert(backend, 1024)
    with patch.object(ws, "dbo_current_ubatch_id", return_value=0):
        first = wrapper(expert)
        first._moe_output.fill_(3)
    with patch.object(ws, "dbo_current_ubatch_id", return_value=1):
        other = wrapper(expert)
        other._moe_output.fill_(5)
    assert torch.all(first._moe_output == 3), "ubatch 1 overwrote ubatch 0"


def test_lock_reuses_existing_resources_but_rejects_unwarmed_owner(backend):
    expert = make_expert(backend, 1024)
    first = wrapper(expert)
    ws.lock_workspace()
    assert wrapper(expert) is first
    with ws.use_workspace_lane(1), pytest.raises(AssertionError, match="locked"):
        wrapper(expert)


def test_reset_does_not_reuse_old_manager_buffers(backend):
    expert = make_expert(backend, 1024)
    first = wrapper(expert)
    first._moe_output.fill_(3)
    ws.reset_workspace_manager()
    ws.init_workspace_manager(torch.device("cuda:0"))
    second = wrapper(expert)
    second._moe_output.fill_(5)
    assert torch.all(first._moe_output == 3)


def test_flag_off_preserves_independent_buffers(backend, monkeypatch):
    monkeypatch.setattr(backend, "_B12X_SHARED_WORKSPACE", False, raising=False)
    a = wrapper(make_expert(backend, 1024))
    b = wrapper(make_expert(backend, 1024))
    a._moe_output.fill_(3)
    b._moe_output.fill_(5)
    assert torch.all(a._moe_output == 3)


def test_decode_bypass_does_not_allocate_b12x(backend):
    expert = make_expert(backend, device="cpu")
    x = torch.ones((3, 2048), dtype=torch.bfloat16)
    output = torch.empty_like(x)

    def cutlass(output, hidden_states, *args):
        output.copy_(hidden_states)

    expert._apply_cutlass = cutlass
    # CUTLASS numerical correctness is unchanged. This test catches allocation
    # before the small-batch return, without compiling an unrelated GEMM.
    with patch.object(expert, "_ensure_wrapper", side_effect=AssertionError("b12x initialized on decode")):
        expert.apply(output, x, None, None, None, None, None, 256, None,
                     None, None, None, None, None, False)
    assert torch.equal(output, x)
