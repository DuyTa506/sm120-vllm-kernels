# Focused pinned PDL protocol ready for a root-granted GPU slot

The first focused baseline has not run. The reviewed protocol is frozen in
`focused_review.json` (SHA256
`c43121ec8632988a3d17004774500e6d7e7889bbe4ecd5387b1ef8009757a0da`).
It covers23 shared serving/workload/FI sources plus7 directory-local driver,
provenance and pin-control sources. All23 shared hashes match the successful
pdl_pin0_1 smoke. Historical smoke launcher/core/reference are unchanged.

Exact first baseline, from the qwen-prefill worktree, only after root grants GPU:

```bash
sg docker -c 'bash kernels/pdl-swapab/run_vllm_pdl_pin_focused.sh pdl_focused_pin0_1 0 pin kernels/pdl-swapab/results/pdl_controlled_reference_v1.json'
```

Estimated9–13minutes including fresh compile/startup, all workloads and cleanup.
Prior pinned smoke startup took173–178seconds; four-case greedy diagnostic added
about200seconds. Five cold NLL rounds add about55–60seconds; the original six
prefill scenarios add further workload/CLI overhead. This is an estimate, not a
measured focused runtime. No candidate run is automatically chained.

Workload is fixed:

- Prefill lengths2048/4096/5000/8192/16384/32768, concurrency1,8 measured prompts
  plus2 warmups each, output1 token.
- Greedy input128/output512, concurrency1/4/16/64 with8/24/96/384 measured
  requests and target-concurrency warmups; temperature0.
- Five cold rounds of the existing ten fixed NLL probes, independent cache salts.
- CONFIGnext/scalar, MTP2, fixed132-entry FI reference, GPU utilization budget0.78.
  Agentic remains the final composed-stack task and is not invoked here.

Reference SHA256 is
`6cfd53ede3b3b99ce7a858ab2c78e067100ce20c697188ebd1da4e9bc317ac27`,
with18 complete identities previously exercised by startup+NLL. **CPU review does
not prove focused lazy-constructor coverage.** Unknown identity, missing singleton
config, late config mutation, source/hash mismatch or bootstrap failure aborts.
No fallback to autotuning is permitted in pin mode. If an unknown identity occurs,
preserve that failed attempt without performance attribution. Only a separately
granted full-focused record can establish a new shared reference, after which
BOTH pin0 and pin1 must restart with fresh caches. Never expand the reference
between paired controls. The record launcher is the same frozen script with mode
`record`; successful completion and canonicalization gates are mandatory.

Explicit env forwarding passes PDL mode, three isolated cache roots, readonly pin
source/bootstrap and exact reference path/hash into the worker subprocesses.
The installed Torch heuristic source hash is guarded by the pin hook. Source
snapshots cover the actual focused driver/helper as well as shared harness files.
Cleanup saves owned container-removal state and a fresh empty-GPU query; failure
sets nonzero cleanup/exit status. Source/FI startup+shutdown/event checks remain.

After a successful run, CPU audit must bind the startup's exact AOT wrappers to
kernel source/config/PTX/cubin evidence (`audit_pdl_artifacts.py`,
`verify_pin_aot.py`). A future paired comparison uses `compare_pin_pair.py
--focused`, enforcing identical frozen protocol, reference, runtime, workload,
actual selected configs and five-round NLL data. The preregistered0.001nat/token
investigation budget is diagnostic only; no quality equivalence is inferred.

Four new frozen-review fixtures plus ten existing pin/protocol/comparator
fixtures pass. Bash/Python syntax and whitespace checks pass. No GPU was used
for this review. The tile64 actual-M3 regression remains documented and is not
expanded or promoted.
