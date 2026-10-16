"""Extract audited HT canonical weights and/or CC packed records from a DLSS-NR DLL."""

from __future__ import annotations

import argparse
import json

from musubi_tuner.dlssnr.dll_models import unpack_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Unpack DLSS-NR weights: HT canonical FP32 and CC lossless packed records.")
    parser.add_argument("--source_dll", "--input", required=True, help="Original DLL containing audited HT and/or CC resources")
    parser.add_argument(
        "--output_dir", "--output", required=True, help="New output directory; multiple resources get HT/CC subdirectories"
    )
    parser.add_argument(
        "--weights", choices=("all", "HT", "CC"), default="all", help="Weights to extract (default: all supported resources)"
    )
    args = parser.parse_args()
    try:
        report = unpack_models(args.source_dll, args.output_dir, weights=args.weights)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    keys = ("output_dir", "roundtrip", "resource_name", "logical_parameters", "canonical_sha256", "record_count", "training_ready")
    summary = {key: report[key] for key in keys if key in report}
    if "models" in report:
        summary["models"] = {name: {key: child[key] for key in keys if key in child} for name, child in report["models"].items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
