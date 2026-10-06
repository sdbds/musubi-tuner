"""Extract an audited DLSS-NR DLL into this project's canonical training format."""

from __future__ import annotations

import argparse
import json

from musubi_tuner.dlssnr.dll_io import unpack_dll


def main() -> None:
    parser = argparse.ArgumentParser(description="Unpack a user-supplied DLSS-NR 310.8.0 DLL into a canonical training directory.")
    parser.add_argument("--source_dll", "--input", required=True, help="Original DLL containing the audited WEIGHTS_HT resource")
    parser.add_argument("--output_dir", "--output", required=True, help="New canonical directory; existing paths are refused")
    args = parser.parse_args()
    try:
        report = unpack_dll(args.source_dll, args.output_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({key: report[key] for key in ("output_dir", "roundtrip", "logical_parameters", "canonical_sha256")}, indent=2))


if __name__ == "__main__":
    main()
