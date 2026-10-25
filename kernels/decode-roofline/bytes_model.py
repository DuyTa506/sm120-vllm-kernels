"""Bytes that one MTP-2 decode step must move from DRAM, from the real checkpoint tensor sizes.

Runs on the host (no torch): parses the safetensors headers.
usage: python3 -I bytes_model.py [snapshot_dir]

One decode step with MTP-2 (num_speculative_tokens=2), per running sequence:
  verify : target model forward over 3 tokens (1 bonus/accepted + 2 drafts)
  draft  : MTP layer pass 1 over the 3 verified positions, pass 2 over 1 token,
           each pass followed by the shared FP8 lm_head on 1 token per sequence
Weights are read once per forward regardless of the number of tokens (if the kernels are ideal),
routed experts only for the distinct experts touched.
"""
import glob, json, re, struct, sys, collections

snap = sys.argv[1] if len(sys.argv) > 1 else glob.glob(
    "/opt/intramind/hf-cache/hub/models--unsloth--Qwen3.6-35B-A3B-NVFP4-Fast/snapshots/*/")[0]
sz = collections.Counter()
for f in sorted(glob.glob(snap + "/*.safetensors")):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        h = json.loads(fh.read(n))
    for k, v in h.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]
        sz[k] = b - a

def tot(pat):
    r = re.compile(pat)
    return sum(v for k, v in sz.items() if r.search(k))

L, LA, LG, E, TOPK = 40, 10, 30, 256, 8
MB = 1e6
dense_gdn = tot(r"language_model\.layers\.\d+\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj|in_proj_a|in_proj_b)\.weight")
dense_attn = tot(r"language_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj\.weight")
router = tot(r"language_model\.layers\.\d+\.mlp\.(gate|shared_expert_gate)\.weight$")
shared = tot(r"language_model\.layers\.\d+\.mlp\.shared_expert\.")
experts_all = tot(r"language_model\.layers\.\d+\.mlp\.experts\.")
per_expert = experts_all / (L * E)
lm_head = tot(r"^lm_head\.")
mtp_dense = tot(r"^mtp\.(fc|layers\.0\.self_attn|layers\.0\.mlp\.(gate|shared_expert))")
mtp_experts_all = tot(r"^mtp\.layers\.0\.mlp\.experts\.")
mtp_per_expert = mtp_experts_all / E

# recurrent state: fp32 (config mamba_ssm_dtype=float32), 32 v-heads x 128 x 128 per GDN layer
state = LG * 32 * 128 * 128 * 4
conv_state = LG * 8192 * 3 * 2          # bf16, width-1 = 3 columns
kv_per_tok = LA * 2 * 2 * 256 * 1       # 10 layers, k+v, 2 kv heads, 256, fp8

def distinct(tokens, k=TOPK, e=E):
    """expected distinct experts for independent uniform top-k routing"""
    return e * (1 - (1 - k / e) ** tokens)

def step(c, ctx=256, overlap=1.0, bw=1.8e12):
    """bytes for one decode step with c sequences; overlap<1 shrinks distinct-expert count"""
    t_ver = 3 * c
    d_ver = min(E, distinct(t_ver) * overlap)
    d_m1 = min(E, distinct(t_ver) * overlap)     # MTP pass 1 sees the same 3 positions
    d_m2 = min(E, distinct(c) * overlap)
    rows = collections.OrderedDict()
    rows["FP8 dense GDN proj (qkv,z,out) + bf16 a/b"] = dense_gdn
    rows["FP8 dense attention q/k/v/o"] = dense_attn
    rows["router + shared-expert gate (bf16)"] = router
    rows["NVFP4 shared experts"] = shared
    rows[f"NVFP4 routed experts ({d_ver:.1f} distinct/layer)"] = d_ver * per_expert * L
    rows["FP8 lm_head (verify)"] = lm_head
    rows["GDN fp32 state read (1 slot)"] = state * c
    rows["GDN fp32 state write (3 slots, one per spec position)"] = 3 * state * c
    rows["conv state r/w"] = conv_state * c * 4
    rows[f"FP8 KV cache read (ctx {ctx})"] = kv_per_tok * ctx * c
    rows["MTP bf16 dense (fc, attn, shared, router) x2 passes"] = 2 * mtp_dense
    rows[f"MTP bf16 routed experts ({d_m1:.1f} + {d_m2:.1f} distinct)"] = (d_m1 + d_m2) * mtp_per_expert
    rows["FP8 lm_head x2 (draft)"] = 2 * lm_head
    return rows

if __name__ == "__main__":
    print(f"per routed expert (gate+up+down, NVFP4 + scales): {per_expert/MB:.3f} MB;"
          f" MTP expert (bf16): {mtp_per_expert/MB:.3f} MB")
    print(f"GDN fp32 state per sequence: {state/MB:.1f} MB; KV per token: {kv_per_tok} B")
    for c, ov in ((1, 1.0), (1, 0.85), (64, 1.0)):
        rows = step(c, overlap=ov)
        s = sum(rows.values())
        print(f"\n== c={c}  (expert overlap factor {ov})")
        for k, v in rows.items():
            print(f"  {v/MB:9.1f} MB  {v/1.8e12*1e6:8.1f} us  {k}")
        print(f"  {s/MB:9.1f} MB total -> {s/1.8e12*1e3:.2f} ms at 1.8 TB/s, "
              f"{s/1.5e12*1e3:.2f} ms at 1.5 TB/s (realistic achievable)")
