"""Opt-in narrow GPU diagnosis of a verifier --failure-dir artifact."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-gpu", action="store_true")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if not args.run_gpu:
        print(json.dumps({"status": "PENDING_GPU"}))
        return
    import torch
    import triton
    import triton.language as tl
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd
    from fused import fused_norm_gate_fp8
    from capture_io import restore_layout

    @triton.jit
    def diagnostic(X, Z, W, Q, S, Y, R, SX: tl.constexpr, SZ: tl.constexpr):
        token = tl.program_id(0)
        head = tl.arange(0, 32)
        channel = tl.arange(0, 128)
        offset = head[:, None] * 128 + channel[None, :]
        x = tl.load(X + token * SX + offset).to(tl.float32)
        variance = tl.sum(x * x, axis=1) * (1.0 / 128.0)
        rstd = tl.rsqrt(variance + 1e-6)
        w = tl.load(W + channel).to(tl.float32)
        z = tl.load(Z + token * SZ + offset).to(tl.float32)
        y = ((x * rstd[:, None]) * w[None, :]) * (z * tl.sigmoid(z))
        y = y.to(tl.bfloat16).to(tl.float32)
        amax = tl.max(tl.max(tl.abs(y), axis=1), axis=0)
        scale = tl.maximum(amax * (1.0 / 448.0), 1.0 / (448.0 * 512.0))
        q = tl.minimum(tl.maximum(y * (1.0 / scale), -448.0), 448.0).to(tl.float8e4nv)
        tl.store(Q + token * 4096 + offset, q)
        tl.store(S + token, scale)
        tl.store(Y + token * 4096 + offset, y)
        tl.store(R + token * 32 + head, rstd)

    torch.set_grad_enabled(False)
    data = torch.load(args.directory / "failure.pt", map_location="cpu", weights_only=True)
    context = json.loads((args.directory / "context.json").read_text())
    if context.get("implementation", "original") != "original":
        raise ValueError("This intermediate probe is specific to the original automatic-layout kernel")
    x, z, w = (restore_layout(data[key], context["specs"][key], device="cuda")
               for key in ("x", "z", "weight"))
    t = x.shape[0]
    q, s = fused_norm_gate_fp8(x, z, w, num_warps=context["warps"])
    dq, ds, y, r = (torch.empty_like(q), torch.empty_like(s), torch.empty_like(x),
                     torch.empty((t, 32), device="cuda", dtype=torch.float32))
    kernel = diagnostic[(t,)](x, z, w, dq, ds, y, r, x.stride(0), z.stride(0), num_warps=context["warps"])
    fy, _, fr = layer_norm_fwd(x.reshape(-1, 128), w, None, 1e-6,
                             z=z.contiguous().reshape(-1, 128), is_rms_norm=True, activation="silu")
    fy = fy.reshape_as(x)
    fr = fr.reshape_as(r)
    def mismatches(a, b, dtype):
        return int((a.contiguous().view(dtype) != b.contiguous().view(dtype)).sum())
    qref = data["q_reference"].cuda()
    bad_q = (q.view(torch.uint8) != qref.view(torch.uint8)).nonzero()
    bad_y = (y.view(torch.int16) != fy.view(torch.int16)).nonzero()
    records = []
    for row, col in bad_q[:12].tolist():
        head, channel = divmod(col, 128)
        records.append({"token": row, "head": head, "channel": channel,
                        "q": float(q[row, col]), "q_reference": float(qref[row, col]),
                        "y": float(y[row, head, channel]), "y_reference": float(fy[row, head, channel]),
                        "rstd": float(r[row, head]), "rstd_reference": float(fr[row, head]),
                        "x": float(x[row, head, channel]), "z": float(z[row, head, channel]),
                        "weight": float(w[channel]), "scale": float(s[row, 0])})
    report = {"context": context, "diagnostic_vs_fused_q": mismatches(dq, q, torch.uint8),
              "fused_vs_saved_q": mismatches(q, data["q_actual"].cuda(), torch.uint8),
              "q_mismatches": len(bad_q), "bf16_norm_mismatches": len(bad_y),
              "rstd_bit_mismatches": mismatches(r, fr, torch.int32),
              "reference_norm_vs_saved": mismatches(fy, data["norm_reference"].cuda(), torch.int16),
              "mismatch_details": records}
    for stage in ("ttgir", "ptx"):
        (args.directory / f"diagnostic.{stage}").write_text(kernel.asm[stage])
    (args.directory / "diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
