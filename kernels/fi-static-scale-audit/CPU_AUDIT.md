# Static M8 comparator ABI audit

Historical context: `moe-nvfp4/README.md` section 4c and `results/small_m_L19.txt` already documented divisor-scale MMA M>=4 versus multiplier direct M1-3. This audit independently verifies that existing contract and corrects the later comparator; it is not a novel kernel fix.

The first numerical probe passed: corrected static input FP4 bytes and SFA exactly match vLLM quantization at both captured offsets and restored input. FP32 errors .134163/.142449 versus CUTLASS .134470/.143675. Original wrong flag remains allzero. See results/fi_static_scale_l19_m8_1/measurement.json; no timing in that run.

The existing M8 `fi_auto_calibrated` comparator is invalid: all three fixtures report FP32 relative L2 exactly 1. Functional and preplanned parity cannot validate two zero outputs. The CPU diagnosis below preceded the separately recorded numerical probe; it is not a claim that the native kernel is faulty.

## Scale derivation

Let `d1=prep.a13=1/a1_gscale`, `d2=prep.a2=1/a2_gscale`; these are dequantization factors. Keep `g1=d1/wgs13`, `g2=d2/wgs2` unchanged.

Installed `flashinfer/cute_dsl/fp4_common.py::quantize_block_fp4` computes `s=E4M3(amax/(6*gs))`, `q=E2M1(x/(s*gs))`; the fast implementation computes the same expression through approximate reciprocals. Thus its argument is **dequantization scale d**, and raw MMA input `q*s` approximates `x/d`. Multiplying FC1 accumulator by `g1=d1/wgs13` restores the intended scale. Static `moe_static_kernel.py:2232` applies alpha to both gate/up before activation. FC2 uses the same divisor helper and `d2`; the final weighted store at line 149 multiplies route weight by down_alpha, rounds that combined factor to BF16, then multiplies BF16 accumulator pairs before the token-major FP32 final reduction. This rounding is another reason not to expect output bit equality with CUTLASS.

By contrast `moe_w4a16_fp4_helpers.py:5420` computes `s=E4M3(amax*gs/6)` and `q=E2M1(x*gs/s)`: its argument is **quantization multiplier 1/d**.

| Backend | Scale supplied | Flag | Effective helper scale | Interpretation |
|---|---:|---:|---:|---|
| Static divisor helper | d1,d2 | false | d1,d2 | Correct candidate |
| Static divisor helper | d1,d2 | true | 1/d1,1/d2 | Existing invalid comparator |
| Direct multiplier helper | 1/d1,1/d2 | false | 1/d1,1/d2 | Existing valid direct multiplier candidate |
| Direct multiplier helper | d1,d2 | true | 1/d1,1/d2 | Existing reciprocal direct candidate |

The static kernel explicitly inverts scales at lines 1471 (input) and 561 (intermediate) when the flag is true. This is not a missing-inversion hypothesis: the inversion plus the imported helper's divisor convention is the mismatch. A unified boolean does not make the backend ABIs identical.

Captured L19 scalar factors: d1=0.01860465109348297, d2=0.009478673338890076. CPU inspection of captured first 8 tokens (1,024 input blocks): wrong-convention SFA has 10 zero blocks, maximum 0.15625; correct SFA has zero zero-blocks, maximum 448. Ignoring quantization error, wrong FC1 scaling attenuates values by d1² (~0.00034613). Subsequent gated activation and wrong FC2 quantization can explain zero output; The later numerical probe confirms input quantization and nonzero corrected output; internal FC1/FC2 intermediate attribution was not instrumented.

## Minimal numerical-only probe

Use existing prepared static Plan tensors/views/workspace and force `static`. Change only `input_scales_are_reciprocal=False`; retain input_global_scale=prep.a13 expanded over experts, fc2_input_scale=prep.a2 expanded, pre-folded w1_alpha=prep.g1, w2_alpha=prep.g2, existing _weight_views, source_format/modelopt and NVFP4 flags. Do not run this setting through automatic dispatch that can select direct micro at another M.

For a fresh functional static comparator without `_weight_views`, retain w1_alpha=1/raw.wgs13 and the same d1 input_global_scale: dispatch folds d1 once. A prepared comparator must not fold g1 again.

Validate forced-static activation; finite/nonzero output; graph/eager/repeat; changed inputs/routes; output against FP32 and valid direct/CUTLASS references. Read packed A/SFA through token_map (route-pair identity), not physical row order. Correct SFA expectation is vLLM scaled_fp4_quant(x,1/d1), mapped to static layout. Count exact scale-byte and payload mismatches separately; fast reciprocal evaluation/rounding ties may require analysis, so algebra alone is not proof of bit equality. Record wrong-flag comparison as diagnostic only. If SFA agrees and output still fails, inspect FC1 alpha and intermediate SFA next. Do not time until numerical gate passes.

## Alternatives inspected

No concrete route/state or output-init defect found: static workspace reserves virtual expert capacity; phase0 resets counts/mapping/counters; token_map retains pair IDs; finalizer overwrites each output. M8 has 64 routed pairs, outside automatic direct `<64` band. N512 needs no intermediate padding. Prepared alpha fold matches functional fold. These checks do not substitute for a GPU intermediate probe.

## Provenance

Installed static, dispatch and helper hashes match the M8 run runtime_sources manifest:

- moe_static_kernel.py: c7b6f24b94d7939cc0eb917ab15cef4c34f3dc12bf75e15fc9dc316ee7327f3f
- moe_dispatch.py: c518e65d6bfd7f08db1e5261e20795fd020e82e681699729171bc2fd5331239a
- moe_w4a16_fp4_helpers.py: 3115cf3e73064cbf2c339d5fd2cec1c49206f34283c281faef2d1c2d290a6c73
- fp4_common.py (current read): a430b3171c7c972a2b98a176e5a47ddcaf36ac71e6231420e961e269d0d045d1

fp4_common was omitted from that run's explicit runtime manifest. It is included in dispatch `_kernel_source_files()` and native disk-cache source invalidation, and the run used a fresh private cache. Thus stale-cache reuse is not the leading explanation, but historical fp4_common identity was not independently archived. Future runs must archive this dependency. CPU tensor/file reads used CUDA_VISIBLE_DEVICES empty; no installed or DP4A source was changed.

## Bucket comparison preparation

`bucket_probe.py` uses forced micro M4 or static M8 with divisor scales and false flag. Micro imports the same fp4_common divisor helper (input lines1119-1152, intermediate1871-1969). Its wrapper adds route compaction before native micro. Both MMA and direct receive safe copied operands from the identical pinned sanitizer, including zeroed inputs on fully invalid tokens and zero-weight valid-ID replacement for -1 slots; sanitation and compaction are included in graph timings. No global hybrid threshold is changed.

Fixtures: real captured offsets0/64, last token padded (M4→3live), all-invalid, restored. The gate permits .01 graph/eager/repeat relative drift for native atomic MMA; direct/CUTLASS exactness is recorded but not claimed for every provider. Allzero padding is exact; full FP32 error threshold .25 is exploratory, not production quality. Only after all pass are full and padded input cases timed in reversed orders (10×50 graph replays); no quantization debugging in the timed calls.

Pinned sanitizer/direct helper copies are in vendor_direct with source SHA attribution; parent DP4A sources remain untouched. Runtime manifest includes recursive installed sm12x Python dependencies plus actual fp4_common, vLLM extension and caller helpers. Native factory selection is Python evidence, not a new CUDA trace.

Commands (separate root grants):
`sg docker -c 'bash kernels/fi-static-scale-audit/run.sh fi_mma_bucket_l19_m4_1 buckets 4'`
`sg docker -c 'bash kernels/fi-static-scale-audit/run.sh fi_mma_bucket_l19_m8_1 buckets 8'`
