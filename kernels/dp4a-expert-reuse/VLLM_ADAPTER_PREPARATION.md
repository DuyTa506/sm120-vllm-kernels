# Zero-copy serving adapter design (conditional on M3 timing)

No vLLM runtime edit, integration or serving run. Current scalar fixed-scale path already retains the exact qf weight/scaling contract:

| qf argument | Current scalar object | condition |
|---|---|---|
| w13/w2 | apply's `w1`/`w2` (current CUTLASS packed operands) | byte-dtype contiguous rows; up/gate order already established before expert processing |
| w13_sf/w2_sf | `self.w1_scale`/`self.w2_scale`, byte views | physical128×4swizzle; do not substitute strided `w*_sf_mma` view |
| g1/g2 | `self._cutlass_g1`/`self._cutlass_g2` | `_process_weights_fixed_scales` lines218–223 already multiplies weight dequant by calibrated activation dequant once |
| a1_gs/a2_gs | first-element views of `self.a1_gscale`/`self.a2_gscale` | verify setup-time scalar or replicated-uniform values, then retain view; no reciprocal/copy in forward |
| output | modular experts output buffer | separate contiguous storage, existing port alias guard |

This mapping uses existing tensor storage, so no Humming-style duplicate weight layout is needed. Actual CPU L19 conversion parity establishes bytes/layout contract against current prep; startup must assert live shape/dtype/stride/scale-uniformity and record data_ptr/storage sharing. All40-layer calibration audit is available but does not substitute for runtime binding checks. Do not bind mutable alphas before original fixed-scale preparation completes.

Port workspace at max8 is approximately4MiB partials +128KiB packed activation +64KiB scale scratch +64B counters, independent of expert weights. A conservative first implementation can own one workspace perlayer (~168MiB total), avoiding concurrent workspace claims. Reuse pooling would be a separate optimization after stream/graph lifetime validation. Every call currently clears counters/output using graph-capturable memset, included in timing; no outputalias permitted.

Eligibility is NOT selected yet. M1 only is numerically/timing screened; M3 is relevant to fixed MTP2; M4/8 straddle FI's internal direct cutoff but current scalar uses CUTLASS below1024. After positive M3/M8 screens, a default-off flag must enter compile_factors, preserve exact other-shape fallback and padded token contract, and bind the extension before graph capture. Invalid expert IDs must remain visible to qf padding handling; do not first replace all padded slots by duplicate expert0. Actual normal-kernel GPU trace and warm-cache OFF→ON→OFF checks are required before serving claims. Retain real vLLM request/context state and actual MTP2 configuration; no standalone fake ForwardContext.
