"""Render one closed-loop clip to a PNG sequence with the FP32 surrogate."""

from __future__ import annotations

import argparse

from musubi_tuner.dlssnr.infer import generate_sequence, load_model, require_surrogate


def main() -> None:
    parser = argparse.ArgumentParser(description="DLSS-NR video inference. Non-reset frames need motion and a history mask.")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--sequence_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--numerics_profile", default="train_surrogate")
    parser.add_argument("--bucket_width", type=int, required=True)
    parser.add_argument("--bucket_height", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    require_surrogate(args.numerics_profile)
    written = generate_sequence(
        load_model(args.model_dir, args.device),
        args.sequence_manifest,
        args.bucket_width,
        args.bucket_height,
        args.output_dir,
        args.seed,
    )
    print(f"wrote {len(written)} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
