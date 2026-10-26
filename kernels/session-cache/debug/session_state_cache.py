# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-session exact-position state retention for hybrid Mamba/GDN models.

Opt-in with ``VLLM_SESSION_STATE_CACHE=1`` (default off). Only active with
prefix caching, Mamba cache mode "align" and the V2 model runner.

Problem: in "align" mode a Mamba/GDN state is only cached at block-aligned
positions (block = 2128 tokens for Qwen3.6-35B-A3B), so the next turn of an
agent session re-prefills everything after the last aligned boundary of the
previous turn, ~1k tokens more than the genuinely new suffix.

Mechanism:

* When a request that carries a ``session_id`` finishes, the scheduler keeps
  - a snapshot of its Mamba state at the exact final position, copied by the
    worker into one fresh block per Mamba group
    (``SchedulerOutput.session_state_snapshots``), and
  - a reference on its last, partially filled attention block(s). Full
    attention blocks before it are already in the prefix cache and are looked
    up by hash on reuse, so an idle session pins only ~4 blocks.
* A later request with the same ``session_id`` whose token ids extend the
  finished sequence (same cache salt / LoRA, checked through the chained block
  hash plus the tail tokens) gets these blocks back from ``get_computed_blocks``
  as a prefix-cache hit of exactly that length. The hit ends inside a block,
  so vLLM's existing partial-hit copy-on-write path gives the request private
  copies and the retained blocks stay intact.

Which position the snapshot represents (MTP / async scheduling):

* No step of the request is in flight when it finishes: the state covers
  ``num_tokens - 1`` tokens (every token except the last sampled one). The
  worker copies the running state at the accepted-token offset, exactly what
  the next decode step would have read.
* With async scheduling one more decode step is usually in flight. Its first
  input token is the request's last sampled token, so after that step the
  state after its first token covers all ``num_tokens`` tokens; the worker
  copies offset 0 of that step's state columns. The step's draft tokens are
  never used. The one case where the align postprocess of that step overwrites
  offset 0 (all drafts accepted and the step ends exactly on a block boundary)
  is skipped.

Spec decoding: the drafter KV of the last retained position may have been
written with a token the new prompt does not have; this can only lower the
draft acceptance of the first decode steps (verification keeps target outputs
exact), the same trade ``disable_eagle_block_drop`` makes.
"""

import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

logger = init_logger(__name__)


def session_state_cache_enabled() -> bool:
    return os.environ.get("VLLM_SESSION_STATE_CACHE", "0") == "1"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class _Entry:
    num_tokens: int  # tokens covered by the retained state (the hit length)
    match_len: int  # a new prompt must equal the finished sequence up to here
    anchor_idx: int  # index into block_hashes anchoring the prefix, -1 if none
    anchor_hash: Any
    tail_start: int
    tail_tokens: list[int]
    cache_salt: str | None
    lora_id: int | None
    # attention spec-group index -> number of leading blocks looked up by hash
    attn_prefix_blocks: dict[int, int]
    # kv cache group id -> retained blocks following the hashed prefix
    attn_blocks: dict[int, list["KVCacheBlock"]]
    # kv cache group id -> snapshot block of the final Mamba state
    mamba_blocks: dict[int, "KVCacheBlock"]
    snapshot_pending: bool
    created: float
    last_used: float
    hits: int = 0
    stats: dict = field(default_factory=dict)


class SessionStateCache:
    def __init__(self, kv_cache_manager: "KVCacheManager", max_model_len: int):
        coordinator = kv_cache_manager.coordinator
        assert isinstance(coordinator, HybridKVCacheCoordinator)
        self.kv_cache_manager = kv_cache_manager
        self.coordinator = coordinator
        self.block_pool = coordinator.block_pool
        self.max_model_len = max_model_len
        self.num_groups = len(kv_cache_manager.kv_cache_config.kv_cache_groups)
        # (group_ids, block_size, spec) per full-attention spec group.
        self.attn_groups: list[tuple[list[int], int, FullAttentionSpec]] = []
        self.mamba_group_ids: list[int] = []
        self.mamba_block_size = 0
        for sg in coordinator.attention_groups:
            if isinstance(sg.spec, MambaSpec):
                self.mamba_group_ids.extend(sg.group_ids)
                self.mamba_block_size = sg.spec.block_size
            else:
                bs = coordinator.single_type_managers[sg.group_ids[0]].block_size
                self.attn_groups.append((list(sg.group_ids), bs, sg.spec))
        self.ttl_s = _env_float("VLLM_SESSION_STATE_TTL_S", 600.0)
        self.max_entries = int(_env_float("VLLM_SESSION_STATE_MAX", 256))
        self.min_free_blocks = int(
            _env_float("VLLM_SESSION_STATE_MIN_FREE", 0.02) * self.block_pool.num_gpu_blocks
        )
        self.log_every = int(_env_float("VLLM_SESSION_STATE_LOG", 100))
        self.entries: OrderedDict[str, _Entry] = OrderedDict()
        # (req_id, use_accepted_offset, {group_id: dst_block_id}) for the worker
        self._pending: list[tuple[str, bool, dict[int, int]]] = []
        self.stats: dict[str, int] = {}
        self._events = 0
        logger.info(
            "Session state cache enabled: attention groups %s, mamba groups %s "
            "(block %d), ttl %.0fs, max %d sessions, min free %d blocks",
            [g for g, _, _ in self.attn_groups],
            self.mamba_group_ids,
            self.mamba_block_size,
            self.ttl_s,
            self.max_entries,
            self.min_free_blocks,
        )

    @classmethod
    def create(cls, scheduler) -> "SessionStateCache | None":
        reason = None
        kvm = scheduler.kv_cache_manager
        if not scheduler.cache_config.enable_prefix_caching:
            reason = "prefix caching is off"
        elif not scheduler.need_mamba_block_aligned_split:
            reason = "no Mamba layers in 'align' cache mode"
        elif not scheduler.vllm_config.use_v2_model_runner:
            reason = "the V1 model runner has no snapshot support"
        elif not isinstance(kvm.coordinator, HybridKVCacheCoordinator):
            reason = "not a hybrid KV cache layout"
        elif scheduler.connector is not None:
            reason = "KV connectors are not supported"
        else:
            for sg in kvm.coordinator.attention_groups:
                mgr = kvm.coordinator.single_type_managers[sg.group_ids[0]]
                ok = (
                    isinstance(sg.spec, MambaSpec) and isinstance(mgr, MambaManager)
                ) or (
                    type(sg.spec) is FullAttentionSpec
                    and type(mgr) is FullAttentionManager
                )
                if not ok:
                    reason = f"unsupported KV cache group {type(sg.spec).__name__}"
                    break
        if reason is not None:
            logger.warning("VLLM_SESSION_STATE_CACHE=1 ignored: %s.", reason)
            return None
        return cls(kvm, scheduler.max_model_len)

    # ------------------------------------------------------------- helpers
    def _count(self, key: str) -> None:
        self.stats[key] = self.stats.get(key, 0) + 1

    def _maybe_log(self) -> None:
        self._events += 1
        if self.log_every > 0 and self._events % self.log_every == 0:
            logger.info(
                "Session state cache: %d sessions, %d retained blocks, stats %s",
                len(self.entries),
                sum(self._entry_num_blocks(e) for e in self.entries.values()),
                dict(sorted(self.stats.items())),
            )

    @staticmethod
    def _entry_blocks(e: _Entry) -> list["KVCacheBlock"]:
        blocks = [b for lst in e.attn_blocks.values() for b in lst]
        blocks.extend(e.mamba_blocks.values())
        return blocks

    def _entry_num_blocks(self, e: _Entry) -> int:
        return len(self._entry_blocks(e))

    def _drop(self, session_id: str, reason: str | None = None) -> None:
        e = self.entries.pop(session_id, None)
        if e is None:
            return
        if reason:
            self._count(f"evict_{reason}")
        if e.snapshot_pending:
            # Not dispatched yet: the snapshot must not land in a reused block.
            ids = {b.block_id for b in e.mamba_blocks.values()}
            self._pending = [
                p for p in self._pending if not ids.intersection(p[2].values())
            ]
        self.block_pool.free_blocks(self._entry_blocks(e))

    # ------------------------------------------------------------- lifecycle
    def maintain(self) -> None:
        """TTL / capacity / memory-pressure eviction, once per schedule()."""
        if not self.entries:
            return
        now = time.monotonic()
        while self.entries:
            sid, e = next(iter(self.entries.items()))
            if now - e.last_used > self.ttl_s:
                self._drop(sid, "ttl")
            elif len(self.entries) > self.max_entries:
                self._drop(sid, "max")
            elif self.block_pool.get_num_free_blocks() < self.min_free_blocks:
                self._drop(sid, "memory")
            else:
                break

    def clear(self) -> None:
        for sid in list(self.entries):
            self._drop(sid, "reset")

    def take_pending(self) -> list[tuple[str, bool, dict[int, int]]] | None:
        if not self._pending:
            return None
        pending = self._pending
        self._pending = []
        for e in self.entries.values():
            e.snapshot_pending = False
        return pending

    def on_finish(self, request: Request) -> None:
        """Called before a finished request's blocks are freed."""
        sid = request.session_id
        if not sid:
            return
        self._count("finished")
        self._maybe_log()
        if request.status not in (
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
        ):
            self._count("skip_status")
            return
        if (
            request.mm_features
            or request.pooling_params is not None
            or request.prompt_embeds is not None
        ):
            self._count("skip_kind")
            return
        n = request.num_tokens
        in_flight = request.num_in_flight_tokens
        frontier = request.num_computed_tokens - in_flight
        if frontier != n - 1:
            # E.g. a stop token in the middle of the accepted drafts: the
            # state also covers tokens that were trimmed from the output.
            self._count("skip_frontier_inflight" if in_flight else "skip_frontier")
            return
        if in_flight == 0:
            num_state_tokens, use_accepted = n - 1, True
        else:
            num_state_tokens, use_accepted = n, False
            if (frontier + in_flight) % self.mamba_block_size == 0:
                self._count("skip_align_boundary")
                return
        if num_state_tokens <= 0 or num_state_tokens >= self.max_model_len:
            self._count("skip_len")
            return

        rid = request.request_id
        attn_blocks: dict[int, list[KVCacheBlock]] = {}
        attn_prefix: dict[int, int] = {}
        for sg_idx, (group_ids, bs, _) in enumerate(self.attn_groups):
            need = cdiv(num_state_tokens, bs)
            k0 = need - 1
            per_group = []
            for gid in group_ids:
                blocks = self.coordinator.single_type_managers[gid].req_to_blocks.get(rid)
                if not blocks or len(blocks) < need or any(
                    b.is_null for b in blocks[:need]
                ):
                    self._count("skip_attn_blocks")
                    return
                k = 0
                while k < need - 1 and blocks[k].block_hash is not None:
                    k += 1
                k0 = min(k0, k)
                per_group.append((gid, blocks))
            attn_prefix[sg_idx] = k0
            for gid, blocks in per_group:
                attn_blocks[gid] = list(blocks[k0:need])

        num_new = len(self.mamba_group_ids)
        # Replace this session's previous entry first (frees its blocks).
        self._drop(sid)
        if self.block_pool.get_num_free_blocks() < num_new + self.min_free_blocks:
            self._count("skip_no_free_blocks")
            return
        snapshot_blocks = self.block_pool.get_new_blocks(num_new)
        retained = [b for lst in attn_blocks.values() for b in lst]
        # +1 ref: the request still holds these, so they never sit in the free
        # queue while the session owns them.
        self.block_pool.touch(retained)

        hbs = self.block_pool.hash_block_size
        match_len = n
        anchor_n = min(match_len // hbs, len(request.block_hashes))
        tail_start = anchor_n * hbs
        now = time.monotonic()
        entry = _Entry(
            num_tokens=num_state_tokens,
            match_len=match_len,
            anchor_idx=anchor_n - 1,
            anchor_hash=request.block_hashes[anchor_n - 1] if anchor_n else None,
            tail_start=tail_start,
            tail_tokens=list(request.all_token_ids[tail_start:match_len]),
            cache_salt=request.cache_salt,
            lora_id=request.lora_request.lora_int_id if request.lora_request else None,
            attn_prefix_blocks=attn_prefix,
            attn_blocks=attn_blocks,
            mamba_blocks=dict(zip(self.mamba_group_ids, snapshot_blocks)),
            snapshot_pending=True,
            created=now,
            last_used=now,
        )
        self.entries[sid] = entry
        self._pending.append(
            (
                rid,
                use_accepted,
                {gid: b.block_id for gid, b in entry.mamba_blocks.items()},
            )
        )
        self._count("retained_no_inflight" if use_accepted else "retained_inflight")
        # DEBUG VARIANT (state check only, not part of the patch): record what
        # the snapshot is supposed to represent.
        dbg = os.environ.get("VLLM_SESSION_STATE_DEBUG_DIR")
        if dbg:
            import hashlib
            import json

            os.makedirs(dbg, exist_ok=True)
            toks = list(request.all_token_ids[:num_state_tokens])
            with open(os.path.join(dbg, "meta.jsonl"), "a") as f:
                f.write(json.dumps({
                    "req_id": rid, "sid": sid, "E": num_state_tokens, "n": n,
                    "in_flight": in_flight, "use_accepted": use_accepted,
                    "num_prompt": request.num_prompt_tokens,
                    "blocks": [entry.mamba_blocks[g].block_id for g in self.mamba_group_ids],
                    "sha": hashlib.sha1(str(toks).encode()).hexdigest(),
                    "t": time.time(),
                }) + "\n")

    def maybe_adopt(
        self,
        request: Request,
        blocks: "KVCacheBlocks",
        num_hit: int,
        shared_prefix_boundary: int,
    ) -> tuple["KVCacheBlocks", int, int]:
        """Replace a shorter prefix-cache hit with the session's exact state."""
        sid = request.session_id
        miss = (blocks, num_hit, shared_prefix_boundary)
        if not sid:
            return miss
        e = self.entries.get(sid)
        if e is None:
            self._count("miss_no_entry")
            return miss
        E = e.num_tokens
        if num_hit >= E:
            self._count("miss_prefix_cache_longer")
            return miss
        if request.num_tokens - 1 < E or request.num_tokens < e.match_len:
            self._count("miss_short_prompt")
            return miss
        lora_id = request.lora_request.lora_int_id if request.lora_request else None
        if request.cache_salt != e.cache_salt or lora_id != e.lora_id:
            self._count("miss_salt")
            return miss
        if e.anchor_idx >= 0 and (
            len(request.block_hashes) <= e.anchor_idx
            or request.block_hashes[e.anchor_idx] != e.anchor_hash
        ):
            self._count("miss_prefix_mismatch")
            return miss
        if list(request.all_token_ids[e.tail_start : e.match_len]) != e.tail_tokens:
            self._count("miss_tail_mismatch")
            return miss

        groups: list[list[KVCacheBlock]] = [[] for _ in range(self.num_groups)]
        for sg_idx, (group_ids, bs, spec) in enumerate(self.attn_groups):
            k0 = e.attn_prefix_blocks[sg_idx]
            prefix: tuple[list[KVCacheBlock], ...] = tuple([] for _ in group_ids)
            if k0 > 0:
                prefix, hit_len = FullAttentionManager.find_longest_cache_hit(
                    block_hashes=request.block_hashes,
                    max_length=k0 * bs,
                    kv_cache_group_ids=group_ids,
                    block_pool=self.block_pool,
                    kv_cache_spec=spec,
                    drop_eagle_block=False,
                    alignment_tokens=bs,
                )
                if hit_len < k0 * bs:
                    self._count("miss_attn_evicted")
                    return miss
            for i, gid in enumerate(group_ids):
                groups[gid] = list(prefix[i]) + list(e.attn_blocks[gid])
        null = self.block_pool.null_block
        for gid, blk in e.mamba_blocks.items():
            groups[gid] = [null] * ((E - 1) // self.mamba_block_size) + [blk]

        e.hits += 1
        e.last_used = time.monotonic()
        self.entries.move_to_end(sid)
        self._count("hit")
        return self.kv_cache_manager.create_kv_cache_blocks(tuple(groups)), E, 0
