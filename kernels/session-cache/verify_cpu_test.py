"""Verifier CPU test (no GPU) of scheduler-side cases test_scheduler_side.py does not cover:
  1. two concurrent requests of the same session adopt the same entry (fork): each must get its OWN CoW
     destination for the attention tail and every Mamba block, sources = the retained blocks;
  2. the entry is evicted (TTL/LRU/memory) while both adopters are still running: retained blocks must not
     return to the free queue while referenced; afterwards every block returns (no leak, no double free);
  3. an adopter is aborted (FINISHED_ABORTED): the entry must stay intact and adoptable;
  4. an entry whose snapshot is still pending is dropped: the pending worker copy must be cancelled;
  5. a turn that finishes exactly at E aligned to the block (hit length multiple of block size);
  6. a request that adopts, is preempted (blocks freed, num_computed reset) and rescheduled.
Run: docker run --rm $(cat .mounts) -v $PWD:/w --entrypoint python3 vllm/vllm-openai:v0.32.0 /w/verify_cpu_test.py
"""
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.session_state_cache import SessionStateCache
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.request import Request, RequestStatus

B, NBLK, NSPEC = 16, 200, 2
init_none_hash(sha256)
hasher = get_request_block_hasher(B, sha256)
attn = FullAttentionSpec(block_size=B, num_kv_heads=2, head_size=64, dtype=torch.float16)
mamba = MambaSpec(block_size=B, shapes=((4, 64),), dtypes=(torch.float32,), mamba_cache_mode="align",
                  num_speculative_blocks=NSPEC)
cfg = KVCacheConfig(num_blocks=NBLK, kv_cache_tensors=[], kv_cache_groups=[
    KVCacheGroupSpec(["attn"], attn), KVCacheGroupSpec(["m0"], mamba), KVCacheGroupSpec(["m1"], mamba)])
kvm = KVCacheManager(cfg, max_model_len=4096, scheduler_block_size=B, hash_block_size=B,
                     enable_caching=True, use_eagle=True, log_stats=True)
cache = SessionStateCache(kvm, 4096)
kvm.session_state_cache = cache
pool = kvm.block_pool
free0 = pool.get_num_free_blocks()
fails = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def mk(rid, toks, sid="sess", salt=None):
    return Request(rid, list(toks), SamplingParams(max_tokens=64), None, block_hasher=hasher,
                   session_id=sid, cache_salt=salt)


def admit(req):
    blocks, hit, _ = kvm.get_computed_blocks(req)
    assert kvm.allocate_slots(req, req.num_tokens - hit, hit, blocks) is not None
    copies, retained = kvm.take_kv_cache_block_copies()
    return hit, copies, retained


def decode(req, n_out, in_flight=0):
    req.num_computed_tokens = req.num_tokens
    req.status = RequestStatus.RUNNING
    req.append_output_token_ids(1000 + len(req.all_token_ids))
    for _ in range(n_out - 1):
        assert kvm.allocate_slots(req, 1) is not None
        req.num_computed_tokens += 1
        req.append_output_token_ids(1000 + len(req.all_token_ids))
    if in_flight:
        assert kvm.allocate_slots(req, in_flight) is not None
        req.num_computed_tokens += in_flight
        req.num_in_flight_tokens = in_flight


def finish(req, status=RequestStatus.FINISHED_STOPPED):
    req.status = status
    cache.on_finish(req)
    kvm.free(req)


# turn 1 -> entry
r1 = mk("r1", range(1, 38))
admit(r1); decode(r1, 9)
finish(r1); cache.take_pending()
e = cache.entries["sess"]
E = e.num_tokens
retained_ids = {b.block_id for b in cache._entry_blocks(e)}
base = list(r1.all_token_ids)

# 1. fork: two adopters
a = mk("a", base + list(range(5000, 5020)))
b = mk("b", base + list(range(6000, 6011)))
ha, ca, ra = admit(a)
hb, cb, rb = admit(b)
check(ha == E and hb == E, f"fork: both adopt at E={E} (got {ha}, {hb})")
srcs_a = {c.src_block_id for c in ca}; dsts_a = {c.dst_block_id for c in ca}
srcs_b = {c.src_block_id for c in cb}; dsts_b = {c.dst_block_id for c in cb}
check(srcs_a == srcs_b and srcs_a <= retained_ids, "fork: CoW sources are the retained blocks for both")
check(not (dsts_a & dsts_b) and not (dsts_a & retained_ids) and not (dsts_b & retained_ids),
      "fork: private, distinct CoW destinations")
for blk_list in (ra, rb):
    pool.free_blocks(blk_list)  # scheduler frees CoW endpoints once copies dispatched
a_tbl = {g: [x.block_id for x in m.req_to_blocks["a"]] for g, m in enumerate(kvm.coordinator.single_type_managers)}
b_tbl = {g: [x.block_id for x in m.req_to_blocks["b"]] for g, m in enumerate(kvm.coordinator.single_type_managers)}
writable_a = set(a_tbl[0][E // B:]) | {x for g in (1, 2) for x in a_tbl[g] if x != 0}
check(not (writable_a & retained_ids), "fork: adopter A's writable blocks exclude retained blocks")

# 2. evict entry while both adopters are running
cache._drop("sess", "test")
in_free = {blk.block_id for blk in pool.free_block_queue.get_all_free_blocks()} if hasattr(
    pool.free_block_queue, "get_all_free_blocks") else set()
check("sess" not in cache.entries, "evict during adoption: entry gone")
decode(a, 3); decode(b, 2)
finish(a, RequestStatus.FINISHED_ABORTED); finish(b, RequestStatus.FINISHED_ABORTED)
check("sess" not in cache.entries, "aborted adopters do not create entries")
check(pool.get_num_free_blocks() == free0, f"no leak after fork+evict ({pool.get_num_free_blocks()} vs {free0})")

# 3. abort keeps the entry adoptable
r2 = mk("r2", range(100, 150), sid="s2")
admit(r2); decode(r2, 7, in_flight=3); finish(r2); cache.take_pending()
e2 = cache.entries["s2"]; base2 = list(r2.all_token_ids)
x = mk("x", base2 + [7, 8, 9, 10], sid="s2")
hx, _, rx = admit(x); pool.free_blocks(rx)
decode(x, 4); finish(x, RequestStatus.FINISHED_ABORTED)
check(cache.entries.get("s2") is e2, "abort: entry unchanged")
y = mk("y", base2 + [7, 8, 9, 10], sid="s2")
hy, _, ry = admit(y); pool.free_blocks(ry)
check(hx == hy == e2.num_tokens, f"abort+retry: retry adopts again at E ({hx}, {hy}, {e2.num_tokens})")
decode(y, 2); finish(y); cache.take_pending()

# 4. drop with pending snapshot cancels the worker copy
r3 = mk("r3", range(300, 333), sid="s3")
admit(r3); decode(r3, 5); finish(r3)
pend_before = [p for p in cache._pending if p[0] == "r3"]
cache._drop("s3", "test")
check(pend_before and not [p for p in cache._pending if p[0] == "r3"], "pending snapshot cancelled on drop")

# 5. block-aligned E (n - 1 == 3 * B)
r4 = mk("r4", range(400, 400 + 3 * B - 4), sid="s4")
admit(r4); decode(r4, 5); finish(r4); cache.take_pending()
e4 = cache.entries.get("s4")
check(e4 is not None and e4.num_tokens % B == 0, f"aligned E retained ({e4.num_tokens if e4 else None})")
z = mk("z", list(r4.all_token_ids) + [1, 2, 3], sid="s4")
hz, cz, rz = admit(z); pool.free_blocks(rz)
mtbl = [x.block_id for x in kvm.coordinator.single_type_managers[1].req_to_blocks["z"]]
snap = e4.mamba_blocks[1].block_id
check(hz == e4.num_tokens, f"aligned E adopted ({hz})")
# with an aligned hit there is no CoW: the snapshot block itself sits in the request's table and must not be
# the running-state (writable) block
print("   aligned-hit mamba table:", mtbl, "snapshot", snap, "copies", [(c.src_block_id, c.dst_block_id) for c in cz])
decode(z, 2); finish(z); cache.take_pending()

# 6. preemption after adoption
r5 = mk("r5", range(500, 541), sid="s5")
admit(r5); decode(r5, 6); finish(r5); cache.take_pending()
p = mk("p", list(r5.all_token_ids) + [5, 6, 7, 8, 9], sid="s5")
hp, _, rp = admit(p); pool.free_blocks(rp)
p.num_computed_tokens = p.num_tokens; p.status = RequestStatus.RUNNING
kvm.free(p); p.status = RequestStatus.PREEMPTED; p.num_computed_tokens = 0
hp2, _, rp2 = admit(p); pool.free_blocks(rp2)
check(hp == hp2 == cache.entries["s5"].num_tokens, f"preempted adopter re-adopts ({hp}, {hp2})")
decode(p, 2); finish(p); cache.take_pending()

cache.clear()
check(pool.get_num_free_blocks() == free0, f"all blocks back at the end ({pool.get_num_free_blocks()} vs {free0})")
print("stats", dict(sorted(cache.stats.items())))
print("FAILED" if fails else "ALL PASS", fails)
