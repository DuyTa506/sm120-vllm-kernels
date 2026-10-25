"""CPU-only test of the scheduler side of the session state cache (no GPU, no model).

Builds a hybrid KV cache manager (1 full-attention group + 2 Mamba "align" groups,
block 16, MTP-style eagle drop on), runs a request through prefill/decode bookkeeping,
finishes it, and checks that a follow-up request of the same session gets an exact
sub-block hit that goes through the copy-on-write path, that mismatching prompts miss,
and that every block returns to the pool afterwards. Run inside the vLLM image:

  docker run --rm $(cat .mounts) -v $PWD:/w --entrypoint python3 vllm/vllm-openai:v0.32.0 /w/test_scheduler_side.py
"""
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.session_state_cache import SessionStateCache
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.request import Request, RequestStatus

B, NBLK, NSPEC = 16, 120, 2
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


def mk(rid, toks, sid="sess", salt=None):
    return Request(rid, list(toks), SamplingParams(max_tokens=64), None, block_hasher=hasher,
                   session_id=sid, cache_salt=salt)


def run(req, n_out, in_flight=0):
    """Prefill then decode n_out tokens (1 per step), the way the scheduler books it."""
    blocks, hit, _ = kvm.get_computed_blocks(req)
    n_new = req.num_tokens - hit
    assert kvm.allocate_slots(req, n_new, hit, blocks) is not None
    req.num_computed_tokens = req.num_tokens
    req.status = RequestStatus.RUNNING
    req.append_output_token_ids(1000 + len(req.all_token_ids))
    for _ in range(n_out - 1):
        assert kvm.allocate_slots(req, 1) is not None
        req.num_computed_tokens += 1
        req.append_output_token_ids(1000 + len(req.all_token_ids))
    if in_flight:
        # async scheduling: one more step already scheduled when the stop arrives
        assert kvm.allocate_slots(req, in_flight) is not None
        req.num_computed_tokens += in_flight
        req.num_in_flight_tokens = in_flight
    return hit


def finish(req):
    req.status = RequestStatus.FINISHED_STOPPED
    cache.on_finish(req)
    kvm.free(req)


# --- turn 1: 37-token prompt + 9 output tokens, no step in flight -> state at 45 tokens
r1 = mk("r1", range(1, 38))
assert run(r1, 9) == 0
n1 = r1.num_tokens
finish(r1)
e = cache.entries["sess"]
assert e.num_tokens == n1 - 1 and cache.take_pending()[0][1] is True, (e.num_tokens, n1)

# --- turn 2 extends turn 1: exact hit at n1 - 1, CoW of the attention tail and both mamba blocks
r2 = mk("r2", list(r1.all_token_ids) + list(range(5000, 5030)))
blocks, hit, _ = kvm.get_computed_blocks(r2)
assert hit == n1 - 1, hit
assert kvm.allocate_slots(r2, r2.num_tokens - hit, hit, blocks) is not None
copies, retained = kvm.take_kv_cache_block_copies()
src = {c.src_block_id for c in copies}
want = {b.block_id for lst in e.attn_blocks.values() for b in lst[-1:]} | {b.block_id for b in e.mamba_blocks.values()}
assert src == want, (src, want)
pool.free_blocks(retained)  # what the scheduler does once the copies are dispatched
r2.num_computed_tokens = r2.num_tokens
r2.status = RequestStatus.RUNNING
r2.append_output_token_ids(7)
for _ in range(4):
    assert kvm.allocate_slots(r2, 1) is not None
    r2.num_computed_tokens += 1
    r2.append_output_token_ids(7)
# finish with a 3-token step in flight -> state at num_tokens (offset-0 snapshot)
assert kvm.allocate_slots(r2, 3) is not None
r2.num_computed_tokens += 3
r2.num_in_flight_tokens = 3
n2 = r2.num_tokens
finish(r2)
e2 = cache.entries["sess"]
p = cache.take_pending()
assert e2.num_tokens == n2 and p[0][1] is False, (e2.num_tokens, n2, p)

# --- mismatching prompts and another salt must miss
bad = list(r2.all_token_ids) + [1, 2, 3]
bad[3] = 99999
r3 = mk("r3", bad)
_, hit3, _ = kvm.get_computed_blocks(r3)
assert hit3 < e2.num_tokens
r4 = mk("r4", list(r2.all_token_ids) + [1, 2, 3], salt="other")
_, hit4, _ = kvm.get_computed_blocks(r4)
assert hit4 == 0
r5 = mk("r5", list(r2.all_token_ids)[:-1] + [42, 43])  # diverges at the last retained token
_, hit5, _ = kvm.get_computed_blocks(r5)
assert hit5 < e2.num_tokens
r6 = mk("r6", list(r2.all_token_ids) + [11, 12])
blocks6, hit6, _ = kvm.get_computed_blocks(r6)
assert hit6 == n2, hit6

# --- stop token trimmed from accepted drafts -> frontier mismatch, no entry
r7 = mk("r7", range(300, 340), sid="s7")
run(r7, 5)
r7.num_computed_tokens += 1  # one more computed than kept tokens
finish(r7)
assert "s7" not in cache.entries and cache.stats.get("skip_frontier") == 1

# --- eviction returns every block
for sid in list(cache.entries):
    cache._drop(sid, "test")
assert pool.get_num_free_blocks() == free0, (pool.get_num_free_blocks(), free0)
print("stats", dict(sorted(cache.stats.items())))
print("OK")
