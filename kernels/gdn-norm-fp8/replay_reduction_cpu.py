"""Replay the original saved failure's norm reduction trees on CPU only."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--token", type=int, default=268)
    parser.add_argument("--head", type=int, default=16)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch

    assert not torch.cuda.is_initialized()
    data = torch.load(args.directory / "failure.pt", map_location="cpu", weights_only=True)
    x = data["x"][args.token, args.head].float().numpy()
    records = []
    for count in (4, 8, 16):
        lanes = x.reshape(128 // count, count)
        # BF16 products here are exactly representable in FP32. Match the
        # emitted local accumulation followed by descending XOR shuffles.
        acc = np.float32(lanes[:, 0] * lanes[:, 0] + lanes[:, 1] * lanes[:, 1])
        for index in range(2, count):
            acc = np.float32(acc + np.float32(lanes[:, index] * lanes[:, index]))
        shift = len(acc) // 2
        while shift:
            acc = np.float32(acc + acc[np.arange(len(acc)) ^ shift])
            shift //= 2
        denominator = np.float32(np.float32(acc[0] / np.float32(128)) + np.float32(1e-6))
        records.append({"values_per_lane": count, "sum_squares": float(acc[0]),
                        "variance_plus_eps": float(denominator),
                        "variance_plus_eps_bits": int(denominator.view(np.uint32)),
                        "cpu_rounded_rsqrt_estimate": float(np.float32(1.0 / np.sqrt(float(denominator))))})
    assert not torch.cuda.is_initialized()
    print(json.dumps({"location": [args.token, args.head], "layouts": records,
                      "cuda_initialized": False,
                      "scope": "CPU float32 sum replay; rsqrt estimate is not GPU rsqrt validation"}))


if __name__ == "__main__":
    main()
