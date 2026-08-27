"""Command-line entry point for configuration validation."""

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an instrument-localization experiment configuration."
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the configuration without running an experiment",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.dry_run:
        print(json.dumps(config.to_dict(), indent=2))
        return 0
    raise SystemExit(
        "Training and inference commands will be added after dataset setup is confirmed. "
        "Use --dry-run to validate the current framework."
    )


if __name__ == "__main__":
    raise SystemExit(main())

