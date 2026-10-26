# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN speculative-decode state replay (``VLLM_GDN_STATE_REPLAY=1``).

The stock fused MTP decode kernel writes the fp32 recurrent state after every
verified position (1 read + T writes per layer and request) although only one of
them survives acceptance. With replay, the verify kernel reads the committed state
once and writes no state. It saves, per token, the three things a replay needs
(``delta``, the normalized ``k`` and the decay). After sampling, ONE kernel for all
layers (:meth:`GdnStateReplay.commit`) folds the accepted tokens into the committed
state with the same fp32 operations in the same order and writes exactly the state
the stock kernel would have left in window slot ``num_sampled - 1`` (plus the
block-boundary state for the prefix cache, derived the way
``postprocess_mamba_fused_kernel`` does). Everything downstream (conv state, align
pre/post copies, ``num_accepted_tokens``) keeps its meaning. Bitwise identical to the
stock path.

A per-layer, per-row ``marker`` set to a device-side step ``epoch`` by the verify
kernel makes the commit a no-op for any row whose verify did not run in this step
(mixed prefill/decode steps and fallbacks keep the stock kernel).
"""

import ctypes
import os
import subprocess

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))


class _VerifyArgs(ctypes.Structure):
    _fields_ = (
        [
            (n, ctypes.c_void_p)
            for n in (
                "mixed_qkv a b a_log dt_bias state_indices cu_seqlens num_accepted "
                "state output_gate norm_weight out rep_delta rep_k rep_decay fold_n "
                "marker epoch"
            ).split()
        ]
        + [
            (n, ctypes.c_int)
            for n in "N W H HV k_heads write_states dt_type norm_w_bf16 sigmoid_gate".split()
        ]
        + [("scale", ctypes.c_float), ("eps", ctypes.c_float)]
        + [
            (n, ctypes.c_int64)
            for n in "mixed_row a_row b_row gate_row state_slot".split()
        ]
    )


def _load_lib() -> ctypes.CDLL:
    path = os.environ.get("VLLM_GDN_REPLAY_LIB", os.path.join(_HERE, "libgdn_replay.so"))
    src = os.path.join(_HERE, "gdn_replay.cu")
    if not os.path.exists(path):
        # JIT fallback: the vllm-openai image ships nvcc.
        out = os.path.join("/tmp", f"libgdn_replay_{os.getuid()}.so")
        cmd = [
            "nvcc", "-O3", "-std=c++17", "-gencode=arch=compute_120,code=sm_120",
            "--use_fast_math", "--shared", "-Xcompiler", "-fPIC", src, "-o", out,
        ]  # fmt: skip
        logger.info("GDN state replay: building %s", out)
        subprocess.run(cmd, check=True)
        path = out
    lib = ctypes.CDLL(path)
    lib.gdn_verify.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.gdn_verify.restype = ctypes.c_int
    lib.gdn_commit.argtypes = [ctypes.c_void_p] * 15 + [ctypes.c_int] * 7 + [ctypes.c_void_p]
    lib.gdn_commit.restype = ctypes.c_int
    return lib


class GdnStateReplay:
    """Process-wide registry of the GDN layers that use replay."""

    def __init__(self) -> None:
        self.lib: ctypes.CDLL | None = None
        self.layers: list = []
        self.epoch: torch.Tensor | None = None
        self._tables: dict = {}
        self._step_groups: list = []
        # Debug: VLLM_GDN_STATE_REPLAY_CHECK=1 runs replay AND the stock kernel in every
        # layer call and compares outputs and states bitwise (eager/piecewise graphs only).
        _chk = os.environ.get("VLLM_GDN_STATE_REPLAY_CHECK", "0")
        self.check = _chk == "1"
        # =2: layer 0 only runs stock (authoritative) + replay and the model-state commit is compared
        # against stock bitwise (tests n / slots / marker / epoch bookkeeping end to end)
        self.shadow = _chk == "2"
        self._shadow = []
        self.sh_steps = 0
        self.sh_rows = 0
        self.sh_bad = 0
        self.sh_armed_bad = 0
        # Below this many requests the stock kernel is faster (the extra commit pass does not pay).
        # Decided per layer call, so FULL graphs captured for small batches keep the stock kernel.
        self.min_reqs = int(os.environ.get("VLLM_GDN_STATE_REPLAY_MIN_REQS", "8"))
        self.checked = 0
        self.bad_state = 0
        self.bad_out = 0

    # ---- layer side -----------------------------------------------------------------
    def register(self, layer, rows: int, num_tokens: int) -> None:
        """Allocate this layer's replay scratch (delta, k, decay, marker)."""
        if self.lib is None:
            self.lib = _load_lib()
        dev = torch.cuda.current_device()
        hv = layer.num_v_heads // layer.tp_size
        h = layer.num_k_heads // layer.tp_size
        kd, vd = layer.head_k_dim, layer.head_v_dim
        layer._replay_rows = rows
        layer._replay_delta = torch.zeros(rows, num_tokens, hv, vd, dtype=torch.float32, device=dev)
        layer._replay_k = torch.zeros(rows, num_tokens, h, kd, dtype=torch.float32, device=dev)
        layer._replay_decay = torch.zeros(rows, num_tokens, hv, dtype=torch.float32, device=dev)
        layer._replay_marker = torch.full((rows,), -1, dtype=torch.int32, device=dev)
        if self.epoch is None:
            self.epoch = torch.zeros(1, dtype=torch.int32, device=dev)
        self.layers.append(layer)

    def verify(
        self, layer, mixed_qkv, a, b, state_indices, cu_seqlens, num_accepted, out,
        output_gate, scale, norm_eps, sigmoid_gate, check=False,
    ) -> bool:  # fmt: skip
        """Stock kernel math, no state writes, saves the replay inputs. False = not applicable."""
        n, w = state_indices.shape
        if n > layer._replay_rows or w > layer._replay_delta.shape[1]:
            return False
        if n < self.min_reqs and not check:
            return False
        state = layer.kv_cache[1]
        p = _VerifyArgs()
        p.mixed_qkv, p.a, p.b = mixed_qkv.data_ptr(), a.data_ptr(), b.data_ptr()
        p.a_log, p.dt_bias = layer.A_log.data_ptr(), layer.dt_bias.data_ptr()
        p.state_indices, p.cu_seqlens = state_indices.data_ptr(), cu_seqlens.data_ptr()
        p.num_accepted, p.state = num_accepted.data_ptr(), state.data_ptr()
        p.output_gate, p.norm_weight = output_gate.data_ptr(), layer.norm.weight.data_ptr()
        p.out = out.data_ptr()
        p.rep_delta, p.rep_k = layer._replay_delta.data_ptr(), layer._replay_k.data_ptr()
        p.rep_decay = layer._replay_decay.data_ptr()
        if not check:  # check mode must not arm the commit
            p.marker, p.epoch = layer._replay_marker.data_ptr(), self.epoch.data_ptr()
        p.N, p.W = n, w
        p.HV = state.size(1)
        p.H = (mixed_qkv.size(1) - p.HV * state.size(2)) // (2 * state.size(3))
        p.k_heads = p.H
        p.dt_type = {torch.float32: 0, torch.bfloat16: 1, torch.float16: 2}[layer.dt_bias.dtype]
        p.norm_w_bf16 = int(layer.norm.weight.dtype == torch.bfloat16)
        p.sigmoid_gate = int(sigmoid_gate)
        p.scale, p.eps = scale, norm_eps
        p.mixed_row, p.a_row, p.b_row = mixed_qkv.stride(0), a.stride(0), b.stride(0)
        p.gate_row, p.state_slot = output_gate.stride(0), state.stride(0)
        r = self.lib.gdn_verify(ctypes.byref(p), torch.cuda.current_stream().cuda_stream)
        if r != 0:
            raise RuntimeError(f"gdn_verify failed ({r})")
        return True

    def check_step(self, layer, mixed_qkv, a, b, state_indices, cu_seqlens, num_accepted, out,
                   output_gate, scale, norm_eps, sigmoid_gate, stock) -> bool:  # fmt: skip
        assert not torch.cuda.is_current_stream_capturing(), "check mode needs non-full graphs"
        state = layer.kv_cache[1]
        n, w = state_indices.shape
        si = state_indices.long()
        src = si.gather(1, (num_accepted.long() - 1).clamp(0, w - 1)[:, None])[:, 0]
        ok_row = src > 0
        s0 = state[src.clamp(min=0)].clone()
        if not self.verify(layer, mixed_qkv, a, b, state_indices, cu_seqlens, num_accepted, out,
                           output_gate, scale, norm_eps, sigmoid_gate, check=True):  # fmt: skip
            stock()
            return True
        out_v = out.clone()
        T = self._build_tables([layer])
        one = lambda t: torch.tensor([t], dtype=torch.int64, device=state.device)  # noqa: E731
        sp, dp, kp, ep, st = (one(state.data_ptr()), one(layer._replay_delta.data_ptr()),
                              one(layer._replay_k.data_ptr()), one(layer._replay_decay.data_ptr()),
                              one(state.stride(0)))  # fmt: skip
        snaps = []
        for t in range(w):
            nn = torch.full((n,), t + 1, dtype=torch.int32, device=state.device)
            r = self.lib.gdn_commit(
                sp.data_ptr(), dp.data_ptr(), kp.data_ptr(), ep.data_ptr(), None, st.data_ptr(), None,
                state_indices.data_ptr(), cu_seqlens.data_ptr(), num_accepted.data_ptr(), nn.data_ptr(),
                None, None, None, None, 0, n, w, 1, T["k_heads"], T["hv"], T["hv"] // T["vpk"],
                torch.cuda.current_stream().cuda_stream)  # fmt: skip
            assert r == 0
            snaps.append(state[si[:, t].clamp(min=0)].clone())
            state[src[ok_row]] = s0[ok_row]  # a commit may overwrite the source slot in place
        stock()
        qlen = (cu_seqlens[1 : n + 1] - cu_seqlens[:n]).long()
        for t in range(w):
            msk = ok_row & (qlen > t) & (si[:, t] > 0)
            if bool(msk.any()):
                cur = state[si[msk, t]]
                bad = (cur.view(torch.int32) != snaps[t][msk].view(torch.int32)).flatten(1).any(1)
                self.bad_state += int(bad.sum())
        self.bad_out += int((out.view(torch.int16) != out_v.view(torch.int16)).flatten(1).any(1).sum())
        self.checked += 1
        if self.bad_state or self.bad_out:
            if self.checked <= 50 or self.checked % 1000 == 0:
                logger.warning("GDN replay CHECK MISMATCH: calls=%d bad_state=%d bad_out=%d",
                               self.checked, self.bad_state, self.bad_out)  # fmt: skip
        elif self.checked % 2000 == 0:
            logger.info("GDN replay check: %d layer calls bitwise identical (states and outputs)", self.checked)
        return True

    def shadow_step(self, layer, mixed_qkv, a, b, state_indices, cu_seqlens, num_accepted, out,
                    output_gate, scale, norm_eps, sigmoid_gate, stock) -> None:  # fmt: skip
        assert not torch.cuda.is_current_stream_capturing()
        state = layer.kv_cache[1]
        n, w = state_indices.shape
        si = state_indices.long()
        src = si.gather(1, (num_accepted.long() - 1).clamp(0, w - 1)[:, None])[:, 0]
        ok_row = src > 0
        s0 = state[src.clamp(min=0)].clone()
        stock()  # authoritative: out and every window slot
        qlen = (cu_seqlens[1 : n + 1] - cu_seqlens[:n]).long()
        R = min(n, 8)  # keep memory bounded: compare the first rows only
        G = torch.stack([state[si[:R, t].clamp(min=0)].clone() for t in range(w)], 1)
        out_stock = out.clone()
        state[src[ok_row]] = s0[ok_row]  # give the replay path the committed state back
        ok = self.verify(layer, mixed_qkv, a, b, state_indices, cu_seqlens, num_accepted, out,
                         output_gate, scale, norm_eps, sigmoid_gate)  # fmt: skip
        if not ok:
            state[src[ok_row]] = s0[ok_row]
            stock()
            return
        self.bad_out += int((out.view(torch.int16) != out_stock.view(torch.int16)).flatten(1).any(1).sum())
        self._shadow.append((G, si.clone(), qlen, ok_row, src.clone(), num_accepted.clone(), layer))

    def _shadow_compare(self, ns: torch.Tensor, n: int) -> None:
        items, self._shadow = self._shadow, []
        nbad_layers = 0
        tot_valid = tot_bad = 0
        first = None
        for G, si, qlen, ok_row, src, acc, layer in items:
            R = G.shape[0]
            si, qlen, ok_row = si[:R], qlen[:R], ok_row[:R]
            nu = torch.minimum(ns[:R].long(), qlen)
            valid = ok_row & (nu > 0)
            idx = (nu - 1).clamp(min=0)
            slot = si.gather(1, idx[:, None])[:, 0]
            valid &= slot > 0
            cur = layer.kv_cache[1][slot.clamp(min=0)]
            exp = G[torch.arange(R, device=G.device), idx]
            bad = (cur.view(torch.int32) != exp.view(torch.int32)).flatten(1).any(1) & valid
            armed = int((layer._replay_marker[:R] == self.epoch).sum())
            self.sh_armed_bad += int(armed != int(ok_row.sum()))
            tot_valid += int(valid.sum())
            nb = int(bad.sum())
            tot_bad += nb
            if nb:
                nbad_layers += 1
                if first is None:
                    r0 = int(bad.nonzero()[0])
                    first = (self.layers.index(layer), r0, int(acc[r0]), int(nu[r0]), int(qlen[r0]),
                             int(src[r0]), int(slot[r0]), si[r0].tolist())  # fmt: skip
        self.sh_steps += 1
        self.sh_rows += tot_valid
        self.sh_bad += tot_bad
        if self.sh_steps <= 6 or tot_bad or self.sh_steps % 200 == 0:
            logger.warning(
                "GDN replay SHADOW step=%d layers=%d rows=%d bad_rows=%d bad_layers=%d (total rows=%d bad=%d "
                "armed_mismatch=%d out_mismatch=%d) first_bad(layer,row,acc,n,q,src,slot,si)=%s",
                self.sh_steps, len(items), tot_valid, tot_bad, nbad_layers, self.sh_rows, self.sh_bad,
                self.sh_armed_bad, self.bad_out, first)  # fmt: skip

    # ---- model-state side -----------------------------------------------------------
    def _build_tables(self, ls: list) -> dict:
        dev = self.epoch.device

        def t(vals, dtype=torch.int64):
            return torch.tensor(vals, dtype=dtype, device=dev)

        k_heads = {x._replay_k.shape[2] for x in ls}
        hv = {x.kv_cache[1].size(1) for x in ls}
        nh = {x._replay_delta.shape[2] for x in ls}
        assert len(k_heads) == len(hv) == 1 and hv == nh, "replay layers must share head counts"
        return dict(
            state=t([x.kv_cache[1].data_ptr() for x in ls]),
            delta=t([x._replay_delta.data_ptr() for x in ls]),
            k=t([x._replay_k.data_ptr() for x in ls]),
            decay=t([x._replay_decay.data_ptr() for x in ls]),
            marker=t([x._replay_marker.data_ptr() for x in ls]),
            stride=t([x.kv_cache[1].stride(0) for x in ls]),
            k_heads=k_heads.pop(),
            hv=hv.pop(),
            vpk=ls[0].num_v_heads // ls[0].num_k_heads,
        )

    def record_step(self, attn_metadata: dict | None, for_capture: bool) -> None:
        """Remember this step's GDN metadata (called from MambaHybridModelState.prepare_attn).

        Hybrid models have several KV-cache groups of GDN layers. Each group has its own block
        table (own spec_state_indices) while layer i of every group shares one physical tensor, so
        the commit must run once per group with that group's indices."""
        self._step_groups = []
        self._shadow = []
        if for_capture or not self.layers or attn_metadata is None:
            return
        by_meta: dict[int, tuple] = {}
        for i, layer in enumerate(self.layers):
            m = attn_metadata.get(layer.prefix)
            if m is None:
                continue
            by_meta.setdefault(id(m), (m, []))[1].append(i)
        self._step_groups = list(by_meta.values())

    def commit(
        self,
        num_sampled: torch.Tensor,
        idx_mapping: torch.Tensor,
        num_computed_post: torch.Tensor | None,
        mamba_block: int,
    ) -> None:
        """Fold the accepted tokens into the committed state of every layer (one launch per group)."""
        groups, self._step_groups = self._step_groups, []
        if not groups or isinstance(num_sampled, int):
            return
        ns = n = None
        for m, idxs in groups:
            # only pure speculative-decode steps run the verify kernel
            if m.num_prefills or m.num_decodes or not m.num_spec_decodes:
                return
            n = m.num_spec_decodes
            si = m.spec_state_indices_tensor[:n]
            if not si.is_contiguous():
                si = si.contiguous()
            key = tuple(idxs)
            T = self._tables.get(key)
            if T is None:
                T = self._tables[key] = self._build_tables([self.layers[i] for i in idxs])
            ns = num_sampled[:n]
            if ns.dtype != torch.int32:
                ns = ns.to(torch.int32)
            im = idx_mapping[:n]
            if im.dtype != torch.int32:
                im = im.to(torch.int32)
            newc = num_computed_post if mamba_block > 0 else None
            if newc is not None and newc.dtype != torch.int32:
                newc = newc.to(torch.int32)
            r = self.lib.gdn_commit(
                T["state"].data_ptr(), T["delta"].data_ptr(), T["k"].data_ptr(), T["decay"].data_ptr(),
                T["marker"].data_ptr(), T["stride"].data_ptr(), self.epoch.data_ptr(),
                si.data_ptr(), m.spec_query_start_loc.data_ptr(), m.num_accepted_tokens.data_ptr(),
                ns.data_ptr(), None, im.data_ptr() if newc is not None else None,
                newc.data_ptr() if newc is not None else None, None, int(mamba_block),
                n, si.shape[1], len(idxs), T["k_heads"], T["hv"], T["hv"] // T["vpk"],
                torch.cuda.current_stream().cuda_stream,
            )  # fmt: skip
            if r != 0:
                raise RuntimeError(f"gdn_commit failed ({r})")
        if self._shadow:
            self._shadow_compare(ns, n)
        self.epoch.add_(1)


_REGISTRY: GdnStateReplay | None = None


def get_replay() -> GdnStateReplay:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = GdnStateReplay()
    return _REGISTRY
