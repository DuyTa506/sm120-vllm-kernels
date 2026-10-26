"""Shared helpers: build the extension, model constants, input generation."""
import glob, os, sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
H, HV, K, V = 16, 32, 128, 128          # linear_num_key_heads, linear_num_value_heads, head dims (config.json)
W = 3                                    # MTP-2: 3 verified positions
CONV_DIM = 2 * H * K + HV * V            # 8192
SCALE = K ** -0.5
EPS = 1e-6                               # rms_norm_eps


import ctypes, subprocess


class _VerifyArgs(ctypes.Structure):
    _fields_ = [(n, ctypes.c_void_p) for n in
                ["mixed_qkv", "a", "b", "a_log", "dt_bias", "state_indices", "cu_seqlens", "num_accepted", "state",
                 "output_gate", "norm_weight", "out", "rep_delta", "rep_k", "rep_decay", "fold_n", "marker", "epoch"]] + \
               [(n, ctypes.c_int) for n in ["N", "W", "H", "HV", "k_heads", "write_states", "dt_type", "norm_w_bf16", "sigmoid_gate"]] + \
               [("scale", ctypes.c_float), ("eps", ctypes.c_float)] + \
               [(n, ctypes.c_int64) for n in ["mixed_row", "a_row", "b_row", "gate_row", "state_slot"]]


class Ext:
    """ctypes wrapper around libgdn_replay.so (built with nvcc, no torch headers needed)."""
    def __init__(self, path):
        self.lib = ctypes.CDLL(path)
        for f in ("gdn_base_decode", "gdn_verify", "gdn_deferred"):
            getattr(self.lib, f).argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            getattr(self.lib, f).restype = ctypes.c_int
        self.lib.gdn_commit.argtypes = [ctypes.c_void_p] * 15 + [ctypes.c_int] * 7 + [ctypes.c_void_p]
        self.lib.gdn_commit.restype = ctypes.c_int

    @staticmethod
    def _p(t):
        return None if t is None else t.data_ptr()

    def _args(self, mixed_qkv, a, b, a_log, dt_bias, state_indices, cu_seqlens, num_accepted, state, output_gate,
              norm_weight, out, scale, eps, sigmoid, rep=None, fold_n=None, marker=None, epoch=None, k_heads=None):
        p = _VerifyArgs()
        for n, t in zip(["mixed_qkv", "a", "b", "a_log", "dt_bias", "state_indices", "cu_seqlens", "num_accepted",
                         "state", "output_gate", "norm_weight", "out"],
                        [mixed_qkv, a, b, a_log, dt_bias, state_indices, cu_seqlens, num_accepted, state,
                         output_gate, norm_weight, out]):
            setattr(p, n, t.data_ptr())
        if rep is not None:
            p.rep_delta, p.rep_k, p.rep_decay = (t.data_ptr() for t in rep)
        p.fold_n = self._p(fold_n)
        p.marker, p.epoch = self._p(marker), self._p(epoch)
        p.N, p.W = state_indices.shape
        p.H, p.HV = H, HV
        p.k_heads = k_heads or H
        p.dt_type = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}[dt_bias.dtype]
        p.norm_w_bf16 = int(norm_weight.dtype == torch.bfloat16)
        p.sigmoid_gate = int(sigmoid)
        p.scale, p.eps = scale, eps
        p.mixed_row, p.a_row, p.b_row, p.gate_row, p.state_slot = (mixed_qkv.stride(0), a.stride(0), b.stride(0),
                                                                   output_gate.stride(0), state.stride(0))
        return p

    def _call(self, fn, p):
        r = getattr(self.lib, fn)(ctypes.byref(p), torch.cuda.current_stream().cuda_stream)
        assert r == 0, f"{fn} failed: {r}"

    def base_decode(self, *a, scale=SCALE, eps=EPS, sigmoid=False):
        self._call("gdn_base_decode", self._args(*a, scale, eps, sigmoid))

    def verify(self, *a, rep, scale=SCALE, eps=EPS, sigmoid=False, marker=None, epoch=None):
        # rep_k is [R, W, H, 128] (per key head)
        self._call("gdn_verify", self._args(*a, scale, eps, sigmoid, rep=rep, marker=marker, epoch=epoch, k_heads=H))

    def deferred(self, *a, rep, fold_n, scale=SCALE, eps=EPS, sigmoid=False):
        self._call("gdn_deferred", self._args(*a, scale, eps, sigmoid, rep=rep, fold_n=fold_n, k_heads=HV))  # rep_k [R, W, HV, 128]

    def commit(self, state_ptrs, delta_ptrs, k_ptrs, decay_ptrs, state_strides, state_indices, cu_seqlens, acc_prev,
               n_new, k_heads=None, marker_ptrs=None, epoch=None, bound_m=None, idx_mapping=None, newc=None,
               mamba_block=0, dst_ptrs=None):
        N, W_ = state_indices.shape
        r = self.lib.gdn_commit(state_ptrs.data_ptr(), delta_ptrs.data_ptr(), k_ptrs.data_ptr(), decay_ptrs.data_ptr(),
                                self._p(marker_ptrs), state_strides.data_ptr(), self._p(epoch),
                                state_indices.data_ptr(), cu_seqlens.data_ptr(), acc_prev.data_ptr(), n_new.data_ptr(),
                                self._p(bound_m), self._p(idx_mapping), self._p(newc), self._p(dst_ptrs), mamba_block,
                                N, W_, state_ptrs.numel(), k_heads or H, HV, H,
                                torch.cuda.current_stream().cuda_stream)
        assert r == 0, f"commit failed: {r}"


def build(verbose=False):
    bd = os.path.join(HERE, "build")
    os.makedirs(bd, exist_ok=True)
    so, src = os.path.join(bd, "libgdn_replay.so"), os.path.join(HERE, "gdn_replay.cu")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        cmd = ["nvcc", "-O3", "-std=c++17", "-gencode=arch=compute_120,code=sm_120", "--use_fast_math", "--shared", "-Xcompiler", "-fPIC",
               "-lineinfo", src, "-o", so]
        if verbose:
            print(" ".join(cmd))
        subprocess.run(cmd, check=True)
    return Ext(so)


def real_params(dt_dtype=torch.bfloat16, w_dtype=torch.bfloat16, layer=0):
    """A_log, dt_bias, norm.weight of a real GDN layer from the checkpoint (HF cache)."""
    from safetensors import safe_open
    snap = glob.glob("/hf/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/*")[0]
    p = f"model.language_model.layers.{layer}.linear_attn."
    with safe_open(snap + "/model-00001-of-00005.safetensors", "pt") as f:
        A = f.get_tensor(p + "A_log").float().cuda()
        dt = f.get_tensor(p + "dt_bias").to(dt_dtype).cuda()
        nw = f.get_tensor(p + "norm.weight").to(w_dtype).cuda()
    return A, dt, nw


class Inputs:
    """One decode step's activations in the strided layout vLLM produces."""
    def __init__(self, qlens, gen, device="cuda"):
        self.qlens = list(qlens)
        L = sum(self.qlens)
        self.L = L
        wide = torch.randn(L, CONV_DIM + HV * V, generator=gen, device=device).to(torch.bfloat16)  # qkvz: [qkv | z]
        self.mixed_qkv = wide[:, :CONV_DIM]
        self.gate = wide[:, CONV_DIM:].view(L, HV, V)
        ba = (torch.randn(L, 2 * HV, generator=gen, device=device) * 2).to(torch.bfloat16)
        self.b = ba[:, :HV]
        self.a = ba[:, HV:]
        cu = [0]
        for q in self.qlens:
            cu.append(cu[-1] + q)
        self.cu = torch.tensor(cu, dtype=torch.int32, device=device)
        self.out = torch.zeros(L, HV, V, dtype=torch.bfloat16, device=device)


def window_indices(B, device="cuda"):
    """Request r owns slots 1+3r .. 3+3r (slot 0 is the null block)."""
    si = torch.arange(1, 1 + B * W, dtype=torch.int32, device=device).view(B, W)
    return si.contiguous()
