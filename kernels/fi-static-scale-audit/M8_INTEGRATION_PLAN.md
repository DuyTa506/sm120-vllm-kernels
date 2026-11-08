# M8 static integration audit — CPU only

A narrow M8 provider is feasible without changing `VLLM_B12X_HYBRID_MIN_TOKENS` or the existing M1/M4 direct provider. This is preparation, not a serving implementation or speed claim.

## Existing callable versus explicit native handle

Installed `moe_dispatch.launch_sm120_static_moe` already accepts caller-owned workspace and `_WeightViews`, live tensors, calibrated scales and output. With M8/topk8 it naturally selects static: 64 routed pairs exceed direct's **actual constant32** and micro's40. The launcher's docstring still says direct `<64`; the constant is authoritative. It does not require writing `_FORCED_BACKEND`. However, it consults that process-global test hook and repeats shape/backend/MAC/factory selection each call. Its scalar expansion and dtype conversions can allocate unless pre-expanded FP32 scales, int32 contiguous IDs and FP32 weights are supplied.

The smallest explicit native handle can compile `_get_static_kernel` once, then call the resulting TVM-FFI callable with the exact runtime tuple at `moe_dispatch.py:1794`. This avoids both global override and per-call dispatch selection. It is a private, source-pinned ABI; do not call just `MoEStaticKernel.kernel`, because the compiled `__call__` launches the cooperative main kernel **and a separate finalizer**. It uses the current TVM-FFI stream; there is no explicit stream argument in the runtime tuple.

Static routing compaction is **inside its main GPU kernel** (phase0/route-pack), not a separate Triton pass. The separate Triton compaction belongs to M4 micro. Do not add a duplicate compaction pass to M8.

## Exact narrow preparation

Reuse the expert's current packed `w13/w2`, `w1_sf_mma/w2_sf_mma`, `_cutlass_g1/_cutlass_g2`, `_input_global_scale/_fc2_input_scale`. Build existing `_WeightViews` without reordering or copying weight tensors beyond existing view/contiguity requirements. Divisor input scales are FP32 expanded `[256]`; alpha factors are already folded once. Set `input_scales_are_reciprocal=False`, fast_math=True, NVFP4/FP4, SiLU.

Allocate the installed static workspace with state_E=weight_E=256,max_rows=64,K2048,N512,topk8. Actual row-count capacity is258 (256 + ceil64/32). Pass **258** as factory state_E to match fake tensor layouts, weight_E256,M8,K2048,N512,topk8,max_rows64,int32 IDs. Use the same hardware-capped static MAC policy as measured: min(110, get_max_active_clusters(1), SM count); do not silently choose a different schedule. Compile outside capture, source-hash guard all device dependencies including fp4_common, and retain its selected config in audit metadata.

The runtime tuple is copied exactly from the pinned dispatcher: live safe input, flattened safe IDs/weights; packed-A view/backing; SFA pointer/backing; route-output scratch; barriers; weight TMA views and backing SFB pointers; counts/maps/virtual routing scratch; scales/alphas; live output; token map/weights. Retain all backing tensors behind raw `data_ptr()` arguments and `_WeightViews` for the handle lifetime. Never retain external warmup input/output tensors as runtime bindings. Call-time tensors must be validated for shape, dtype, device, contiguity and non-overlapping output, as in LiveDirect.

The same existing sanitizer runs on each call. It changes -1 padding to safe valid expert IDs with zero route weight, and zeros private input for fully invalid tokens. It preserves valid routing and weights. Malformed IDs must be handled through its existing diagnostics contract. Native static directly indexes expert arrays and must not receive -1. Sanitation is included in any performance claim.

## Scratch ownership and graph lifecycle

Use `WorkspaceManager.get_persistent_resource` keyed by provider/version, expert owner and M8. Different expert owners need distinct live scratch under the current conservative ownership policy. Shared sequential scratch across layers is an untested future optimization, not part of this integration.

Build, warm sanitizer/JIT and execute at least one complete call before graph capture; reject late first creation inside capture. No tensor allocation, CPU routing reads, device synchronization or backend override inside captured execution. The main kernel resets routing counts/maps/claim counters each launch; epoch barriers initialize once and persist; the finalizer overwrites output. Preserve both cooperative main launch and finalizer on the same stream. Do not introduce a zero-output operation unless the existing compiled entrypoint requires one (it does not).

Required new verification before serving: three distinct input/output pointer sets, changed route/input graph replay, full/padded7/all-invalid/restore, cross-owner interference, graphs across M1/M4/M8 composition, and real graph activation showing static main + finalizer. Existing M8 numeric/timing checks prove fixed-buffer graphs, not these live-pointer and multi-owner cases. Do not widen beyond actual M8 geometry or alter the parent hybrid threshold.

Cleanup follows owner/manager lifetime after graphs stop using the handle. Dropping a Python temporary is insufficient if graphs or manager retain resources; do not reuse/finalize scratch while a captured graph can replay. Serving setup should report measured live allocation and peak before any promotion.

## Memory estimate (source-derived, not observed serving allocation)

Unique static workspace allocations at M8:

| Buffer group | Bytes |
|---|---:|
| packed input,258×64×1024 | 16,908,288 |
| SFA,258×128×128 | 4,227,072 |
| route output scratch,64×2×2048BF16 | 524,288 |
| counts/maps/weights/barriers/virtual scratch and compact-ID array | 139,324 |
| Static workspace subtotal | 21,798,972 (20.789MiB) |
| Sanitizer private input/IDs/weights/errors | 33,312 |
| Two expanded divisor scale planes | 2,048 |

The public allocator also creates direct-only planes (69,120bytes at M8) even though static does not use them. A handle retaining the entire workspace keeps those extra planes; a native handle retaining only bound tensors can release them and the unused1,024byte compact-ID array. First implementation should prefer correct lifetime over trimming tiny buffers.

For comparison, a native direct handle retains about67,072bytes of intermediate/barriers plus2,048bytes of prepared multiplier scales and33,312bytes of sanitizer at M8; it does not retain the large static packed/SFA planes after its temporary Plan is released. Thus static adds roughly20.7MiB per expert owner. Forty independent M8 owners would add roughly0.81GiB over direct-sized scratch, before allocator alignment, graph pools, temporary setup peak and existing weight-view storage. No extra packed weight copy is inherently required. This memory cost must be measured against KV-cache capacity.

## Difficulty and recommendation

Moderate and self-contained: a source-pinned native handle plus one exact-M8 opt-in branch can reuse the tested sanitizer, owner resource manager and existing weights. The principal work is lifetime/live-pointer verification and measured memory; the math and kernel already passed this L19 screen. There is no need to port a new scheduler or change FlashInfer globally. First finish the ongoing M1/M4 serving evidence, then decide whether M8 occurs often enough to justify its extra scratch and another paired serving experiment.
