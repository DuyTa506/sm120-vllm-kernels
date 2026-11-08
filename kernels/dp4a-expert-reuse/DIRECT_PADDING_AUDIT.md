# FI direct padding adapter, CPU preparation

Pinned installed `sources/moe_direct_micro_kernel.py` SHA256
`9b18f54588c3f24cab74dff07545f323e2af8235a4532233655a5a75498e89cc`.
Native FC1 lines2206–2220 and FC2 lines1205–1219 (M1:766–781) dereference
expert arrays without negative-ID guards. Raw-invalid invocation remains unsupported.

Duplicate expert IDs are not expert-owner aliases on this direct implementation:
FC1 lines2195–2203 chooses token and top-k slot from the route index; its
intermediate storage is per route slot. FC2 lines1179/1205/1222 reads
`token_inter_base + kk*128`, loops all `kk`, and applies that slot's router weight.
M1 uses `kk*128` at791. `DirectNative` explicitly sets both sharing flags false
and activation SiLU, matching the validated direct Plan. It does not use DP4A's
expert-owner deduplication. The GPU diagnostic nevertheless includes all-eight
slots expert0 to test this source-derived duplicate legality independently.

`direct_sanitize.py` always launches one Triton kernel with one program per token:

- Writes private int32 IDs and FP32 weights, mapping padding to safe expert0 and
  zero weight. This makes duplicates, whose direct-only slot semantics are above.
- Copies BF16 hidden input for tokens with valid routes. Fully invalid tokens get
  zero input even if original hidden/weights contain NaNs; no posthoc output mask.
- Rewrites every buffer and per-token malformed-ID counter every call. Legal
  IDs are -1 or0..255. Malformed IDs are memory-sanitized but *not semantically
  accepted*: the diagnostic must reject their nonzero counter; no host read is
  inserted in the production-shaped launch. Upstream route contract must hold.

Private sanitizer storage is `4164*M` bytes (33,312 bytes at M8), plus existing
per-shape FI workspace/epoch storage. Setup allocation/JIT happens before graph
capture. Sanitization is included in every `SanitizedCall`, including timing.
No process-global forced-backend mutation occurs in the native prepared handle.

Exact Triton-interpreter execution on CPU passed20 fixtures across M1/3/4/8,
including partial padding with NaN invalid weights, wholly padded NaN hidden,
malformed -2/256 IDs, and restoration. CUDA remained uninitialized. These are
CPU semantics checks, not GPU numerical proof. Fourteen host unit tests pass.

Pending focused diagnostic includes reciprocal Plan, multiplier Plan and explicit
native handle, each with its own sanitizer/workspace. It revisits M1/3/4/8 graphs,
requires fresh eager/repeat equality, raw-valid Plan parity, FP32 sanity and exact
padded-token zeros. Actual CUDA profiling follows gates. No GPU launch authorized
yet; source review must precede the requested focused M1 batched run.
