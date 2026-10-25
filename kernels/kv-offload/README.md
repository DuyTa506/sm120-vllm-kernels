# CPU KV offloading for idle agent sessions (vLLM v0.32.0, Qwen3.6-35B-A3B hybrid)

STATUS: draft, measurements pending (see sections marked TODO)

## 1. Does vLLM v0.32 offloading support this hybrid model? (code reading, verified by a run)

**Yes, the native backend handles both attention KV and GDN (Mamba-type) state.** Nothing is missing for the
basic "keep idle sessions in host RAM" use case. Source refs are relative to `vllm/` in the image.

| piece | where | what it does |
|---|---|---|
| `--kv-offloading-size N --kv-offloading-backend native` | `config/vllm.py:1115` | selects `OffloadingConnector` (or `SimpleCPUOffloadConnector` with `VLLM_USE_SIMPLE_KV_OFFLOAD=1`), `cpu_bytes_to_use = N GiB` |
| hybrid allocator (HMA) stays on | `config/vllm.py:2170`, `offloading_connector.py:53` | both native connectors subclass `SupportsHMA`, so the hybrid KV cache manager is not disabled |
| which groups are offloaded | `offloading/config.py:36` | `prefix_cacheable_group_ids`: attention group **and the 3 GDN groups** (align mode makes them prefix-cacheable) |
| Mamba page copy | `offloading/worker.py:126` | GDN pages are copied as raw bytes (unpadded page = conv state + SSM state of that layer) |
| attention store | `offloading/scheduler.py:_build_store_jobs` | write-through: every completed 2128-token chunk of the attention group is copied D2H as it is computed |
| Mamba store | `offloading/scheduler.py:_build_aligned_boundary_store_jobs` | the Mamba state is stored only at boundaries the GPU `MambaManager` hands off (`boundary_state_offloads`, `single_type_kv_cache_manager.py:1961-2006`): the retained block-aligned checkpoints (with default retention: the prompt's last 2128-aligned boundary) |
| lookup | `offloading/scheduler.py:147, 752` | attention: longest prefix of stored chunks; Mamba: treated as a 1-chunk sliding window, i.e. the hit end must have a stored state; hit rounded down to the 2128 block (`resolve_mamba_align_size`) |
| load | `kv_offload/cpu/gpu_worker.py` | async H2D on side CUDA streams from a pinned `/dev/shm` mmap; the request waits in `WAITING_FOR_REMOTE_KVS` while other requests keep running |
| MTP | `offloading/scheduler.py:254` | group 3 (the attention group, which also holds the MTP layer) is flagged EAGLE/MTP: its volatile last chunk is not stored while decoding (stored at finish) |
| partial tail (state at a sub-block boundary) | `offloading/scheduler.py:305` | only if `tokens_per_hash < 2128` (`--prefix-match-unit`), blocks_per_chunk == 1 **and no EAGLE/MTP group**. With MTP on it is therefore **off**: CPU restores land on 2128 boundaries even when the GPU cache uses a finer `--prefix-match-unit` |

Startup log of the smoke run (`logs/smoke_native.server.log`): `kv cache group sizes (2128, 2128, 2128, 2128)`,
`Created mmap file /dev/shm/vllm_offload_<id>.mmap (103.07 GB)`, `KV offloading: EAGLE/MTP draft attention groups [3]
detected`. Pinning 103 GB (cudaHostRegister) adds ~105 s to server start.

**LMCache** (`--kv-offloading-backend lmcache`): v0.32 maps it to `LMCacheMPConnector`, and the lmcache 0.5.6 wheel in the
image ships an external `LMCacheMPConnector` that subclasses `SupportsHMA` and contains Mamba-align handling
(`lmcache/integration/vllm/lmcache_mp_connector.py:371 validate_mamba_step_alignment`). It needs a separate LMCache
server process (or `lmcache.mp.autostart`), `--kv-offloading-size` is not propagated (size goes to the server's
`--l1-size-gb`), and its chunk size must be a multiple of the 2128-token block. TODO result.

## Sizes (measured from transfer counters)

One 2128-token block of one KV cache group = 11 layers x 2.167 MB = 23.84 MB (10 attention layers + the MTP layer; the
3 GDN groups are padded to 11 layers, "Add 3 padding layers"). So in host RAM:
- attention KV: 23.84 MB / 2128 tokens = **11.2 KB per token** (a 100k-token session: ~1.1 GB)
- one GDN checkpoint (3 groups): **71.5 MB**
- `--kv-offloading-size 96` (103 GB) holds ~4,320 blocks, ~9.2M tokens of attention KV (2.3x the GPU's 3.95M)
