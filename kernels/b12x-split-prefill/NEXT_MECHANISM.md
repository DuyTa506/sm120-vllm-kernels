# What this screen supports next

No further GPU request for the initial split candidate. Native arithmetic/domain
correctness is sufficiently understood for an exploratory screen; complete-call
performance is consistently worse at both measured shapes.

A precise *conditional* kernel hypothesis is replacing the FC2 compute/copy
pipeline with an M128, warp-specialized TMA mainloop while retaining the FP32
accumulator boundary and original `(g2 * route_weight) * accumulator` → BF16
atomic epilogue. The pinned sm120_nvfp4_ops grouped implementation supplies an
existing384-thread/256-math-thread, three-stage pipeline reference; it is not a
ready compatible replacement because its published dispatcher hardcodes FP16
output and has per-invocation descriptor costs.

A tractable transplant would reuse this device count/tile domain, change phase1's
intermediate **scale storage only** from `[slice,row,8bytes]` to F8_128x4 (packed
payload unchanged), and read original prepared down weights/scales. Customize the
new kernel epilogue before its FP16 conversion to perform existing g2/topk scaling
and BF16 atomic scatter. Prove intermediate payload/scale decoding and local
weighted contributions first; include descriptor refresh and new layout stores
in the complete-call measurement. Preserve the existing candidate as control.

Data supports FC2 as a substantial cost center (~215µs at4096), but does not show
that copy stalls rather than atomics or arithmetic cause it. Replacing FC2 alone
would need an implausibly large gain to recover the observed~205µs complete-call
gap. Therefore this is a reusable grouped-mainloop research mechanism, not an
immediate optimization claim or recommended expensive port. First inspect
resource/stall evidence if that direction is selected. A cheaper publisher-only
change also has no demonstrated path to recovering the full gap.

Prioritize the already active monolithic workspace/tile study. It attacks actual
padding/dispatch without surrendering the existing fusion. Persistence, masked
valid rows and occupancy split scheduling already exist there; adding their
names does not create a new optimization.
