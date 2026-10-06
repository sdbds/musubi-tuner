"""Quantize a full checkpoint or checked LoRA merge into a copy of an original DLL."""

from __future__ import annotations

import argparse
import json
import logging

from musubi_tuner.dlssnr.dll_io import pack_dll


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pack DLSS-NR canonical weights into a new DLL; native loading is not certified.")
    parser.add_argument("--template_dll", required=True, help="Original audited DLL; never modified")
    parser.add_argument(
        "--model_dir", "--input", required=True, help="Full canonical checkpoint, or the matching base for --merge_lora"
    )
    parser.add_argument("--output_dll", "--output", required=True, help="New DLL path; also writes <path>.report.json")
    parser.add_argument(
        "--mix", type=float, default=1.0, help="Template/trained interpolation in [0,1]; 0=template, 1=trained (default)"
    )
    parser.add_argument(
        "--strength",
        "--multiplier",
        type=float,
        default=1.0,
        help="Scale ALL numeric weights after mixing (default 1); not a visual effect strength. Opaque bytes are unchanged.",
    )
    parser.add_argument(
        "--merge_lora", "--lora_weight", help="DLSS-NR adapter.safetensors to merge into --model_dir before packing"
    )
    parser.add_argument("--lora_multiplier", type=float, default=1.0, help="Scale only the LoRA delta before mixing (default 1)")
    return parser


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        report = pack_dll(
            args.template_dll,
            args.model_dir,
            args.output_dll,
            mix=args.mix,
            strength=args.strength,
            merge_lora=args.merge_lora,
            lora_multiplier=args.lora_multiplier,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    summary = {key: report[key] for key in ("output_dll", "output_sha256", "whole_dll_byte_identical", "totals")}
    summary["report"] = str(args.output_dll) + ".report.json"
    print(json.dumps(summary, indent=2))
    if report["totals"]["blended_changed_values"] and not report["totals"]["exported_changed_values"]:
        logging.warning("All requested changes rounded back to the template weights; the DLL is unchanged.")
    logging.warning("Native loading/image quality is unverified; no PE checksum or signature repair is performed.")


if __name__ == "__main__":
    main()
