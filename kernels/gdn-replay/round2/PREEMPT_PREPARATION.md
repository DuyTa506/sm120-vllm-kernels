# Preemption follow-up (CPU preparation only)

Shadow3 passes actual boundary/state checks but preemption metric delta is zero. A dedicated pressure run is still required before timing.

Observed startup: 1,996 GPU blocks, 2,176-token attention/Mamba block size, align mode, MTP2, 30FP32 GDN layers. Reported max concurrency26.263 at max context131072 is consistent with76 resident pool blocks per max-length request. This does not uniquely identify the actual cache groups. `cache_geometry.py` is a prepared startup-only serializer of actual runner.kv_cache_config groups, per-group specs/page sizes and max-request resident blocks; it is not wired into the frozen composition yet.

Installed `MambaSpec.max_memory_usage_bytes` uses `2 + num_speculative_blocks + num_prefill_checkpoint_blocks` pages in align mode. Full-attention pages grow with sequence length. Thus choosing a cap by total token capacity alone is insufficient: the scheduler may admit fewer requests and never preempt. Root must review the actual group/spec record before selecting a cap.

Candidate pressure recipe after geometry confirmation: keep fixedMTP2 and all current numerical patches; set preempt-only max-model-len16384, use short128-token prompts with long continuations crossing2176 at concurrency16. A cap near50 (not selected yet) could allow a small initial cohort and force eviction when attention pages grow. Cap64 may leave enough slack that no eviction occurs. Both are hypotheses, not measured admission geometry.

Before launch, generate the exact stress prompts using existing deterministic seeds, tokenize with the actual checkpoint tokenizer, and verify max(prompt_tokens + requested_generation + draft allowance) <=16384. Save prompt/token hashes and counts. Do not rely on the stress filename/word-count parameter as an exact token bound. The benchmark client lengths and completion guards must remain fixed and verified. Actual finite, nondecreasing preemption-counter delta >0 remains mandatory; predicted pressure is not coverage.

Prepared exploratory configuration is now explicit: KVBLOCKS50, max-model-len16384, same32 boundary requests (128+2304), then unchanged128-request stress. Startup must report exactly one full-attention and three Mamba groups, block2176, MTP2 speculative blocks and one internal-prefill checkpoint block per Mamba group. The latter comes from current build QwenGatedDeltaNetAttention.get_kv_cache_spec (internal checkpoint override), yielding2+2+1=5 resident blocks/group. Startup records actual geometry then rejects any mismatch before requests. Capacity hypothesis:49 usable pool blocks, initial cohort floor(49/(1+3*5))=3; growth after2176 needs3*(2+3*5)=51>49. This is an upper-budget model; allocator behavior and prefix reuse may alter admissions, so observed preemptiondelta remains decisive.

Exact CPU token preflight now executes the actual stress source prompt/body assignments and tokenizes all128 with pinned checkpoint tokenizer. Maximum prompt+generation+3 draft allowance is5213, below16384;37 aborts match completed shadow3 stress. Generator source SHA, prompt/token hashes and CUDAfalse are saved in stress_tokens_cpu.json. Runner fails if generator source changes.

Observed outcome of cap50/2304-output experiment: actual geometry guard passed, but four requests were admitted and no eviction occurred. All workload and numerical checks passed; preemption gate correctly failed. The maximum-resident-budget arithmetic did not predict actual per-step residency/admission. A future reviewed experiment could keep cap50 and use8 requests at concurrency8 with6656 output tokens (128+6656 traverses three2176-token boundaries). This increases full-attention residency while holding the Mamba window bounded. It is a new pressure hypothesis, not a selected or granted run.

CPU-prepared command (requires explicit new GPU lease):

```
sg docker -c 'PREEMPT_GROWTH=1 KVBLOCKS=50 bash kernels/gdn-replay/round2/run.sh replay_r2_preempt_growth1 preempt'
```

The opt-in profile writes boundary_config.json and requires all8 exact6656-token outputs. Original shadow/default preempt workload remains32×2304. Allocation observer samples raw HTTP metrics once per second (KV usage, running/waiting requests, cumulative preemptions) into allocation_trajectory.jsonl; this is an instrumented correctness workload, not timing. Old negative preempt1 evidence is retained. Actual sampler and14 CPU guard tests pass.

### Actual allocator audit after preempt1

The earlier16-block/request arithmetic is a conservative peak budget, not actual residency. Installed `single_type_kv_cache_manager.py:1714–1775` uses main-model tokens in align mode, caps new physical Mamba blocks at `1 + partial_hit + checkpoint`, and adds two speculative blocks only at first allocation. For a128-token new prompt, no internal boundary checkpoint exists: each of three Mamba groups initially owns3 physical blocks. With one attention block this is10/request; four requests consume40 of49 usable pool blocks, explaining observed4 admissions. Logical table length includes null placeholders and must not be interpreted as physical residency.

At lines1800–1861, existing speculative scratch blocks are relocated and new boundary state storage is allocated; skipped old state blocks are freed by `remove_skipped_blocks`. In ordinary decode, each Mamba group is approximately3 resident blocks, temporarily4 around a boundary. The5-block `MambaSpec.max_memory_usage_bytes` bound also allows checkpoint/previous-state peaks and is not constant. Prefix sharing, copy-on-write and checkpoint steps can alter this simple accounting.

For four concurrent requests, steady attention growth gives approximately40/44/48/52 physical blocks as total sequence length crosses2176/4352/6528; transient boundary allocation can run out earlier. Thus8 requests/c8 with128+6656 tokens is a stronger source-backed pressure case than2304 outputs: four surviving overlapping requests cannot all retain the final steady52-block state in a49-block pool. Completion skew, prefix reuse and admission timing still prevent claiming guaranteed observed eviction from this estimate alone. Expected trajectory: initially about4running/4waiting, increasing or sawtooth KV usage at boundaries, then a real cumulative preemption increment and running/waiting changes. The1Hz observer can miss individual transitions; the final metric gate remains authoritative.

`kv_cache_manager.py:546–579` removes skipped blocks, sums coordinator requirements, and returnsNone when free capacity cannot satisfy them. Running-request allocation failure reaches scheduler.py775–848 preemption/retry logic; waiting-request failure around1256–1282 stops admission instead. Queueing alone is not preemption. No allocator/runtime/gate changes were made by this audit; the growth command remains awaiting an explicit lease.
