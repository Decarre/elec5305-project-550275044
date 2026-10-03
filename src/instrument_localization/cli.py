"""Command-line interface for validation, training and frozen evaluation."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .config import load_config


def _legacy_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an instrument-localization experiment configuration."
    )
    parser.add_argument("--config", type=Path, required=True, help="YAML configuration")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print configuration")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate and print a YAML config")
    validate.add_argument("--config", type=Path, required=True)

    train = subparsers.add_parser("train", help="Train from feature/clip-label NPZ files")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--train-data", type=Path, required=True)
    train.add_argument("--validation-data", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--device", choices=("cpu", "cuda"))

    evaluate = subparsers.add_parser(
        "evaluate", help="Evaluate a checkpoint using its frozen validation thresholds"
    )
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--output-json", type=Path, required=True)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    # Preserve the original documented dry-run interface.
    if arguments and arguments[0].startswith("-"):
        args = _legacy_parser().parse_args(arguments)
        config = load_config(args.config)
        if args.dry_run:
            print(json.dumps(config.to_dict(), indent=2))
            return 0
        raise SystemExit("Use --dry-run or one of the validate/train/evaluate commands.")

    args = build_parser().parse_args(arguments)
    if args.command == "validate":
        print(json.dumps(load_config(args.config).to_dict(), indent=2))
        return 0

    from .training import evaluate_checkpoint, load_npz_dataset, train_arrays

    if args.command == "train":
        config = load_config(args.config)
        train_features, train_labels = load_npz_dataset(args.train_data)
        validation_features, validation_labels = load_npz_dataset(args.validation_data)
        summary = train_arrays(
            train_features,
            train_labels,
            validation_features,
            validation_labels,
            config,
            args.output_dir,
            patience=config.early_stopping_patience,
            device=args.device,
        )
        print(json.dumps(summary, indent=2))
        return 0

    features, labels = load_npz_dataset(args.data)
    metrics = evaluate_checkpoint(args.checkpoint, features, labels, args.device)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
