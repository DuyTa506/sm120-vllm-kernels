"""CPU cost of request block hashing at prefix-match unit 16 vs 2128 (runs inside the vLLM image, no GPU).
Each agent turn is a new request, so the whole prompt is hashed every turn."""
import random, time
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash

class R:  # minimal request stand-in
    def __init__(s, toks):
        s.all_token_ids = toks; s.num_tokens = len(toks); s.block_hashes = []
        s.mm_features = []; s.lora_request = None; s.cache_salt = "x"; s.prompt_embeds = None
        s.num_prompt_tokens = len(toks)
init_none_hash(sha256)
for n in (30000, 126000):
    toks = [random.randrange(150000) for _ in range(n)]
    for unit in (2128, 64, 16):
        h = get_request_block_hasher(unit, sha256)
        best = 1e9
        for _ in range(3):
            r = R(toks); t = time.perf_counter(); hs = h(r); best = min(best, time.perf_counter() - t)
        print(f"prompt {n:6d} unit {unit:5d}: {len(hs):5d} hashes, {best*1e3:7.1f} ms")
