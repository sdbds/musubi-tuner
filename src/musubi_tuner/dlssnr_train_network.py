"""DLSS-NR LoRA training. The full-training entry rejects this config."""

from __future__ import annotations

import argparse
import logging

from musubi_tuner.training.dlssnr_trainer import train_lora_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DLSS-NR LoRA on paired frames.")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Exact adapter/optimizer state directory.")
    parser.add_argument("--development_smoke", action="store_true", help="Explicitly allow unvalidated experimental runs.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    train_lora_from_config(
        args.config_file, max_steps=args.max_train_steps, resume=args.resume, development_smoke=args.development_smoke
    )


if __name__ == "__main__":
    main()
