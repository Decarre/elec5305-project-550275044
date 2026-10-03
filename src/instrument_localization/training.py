"""Reproducible array-based training, validation tuning and checkpoint reload."""

import csv
import json
import random
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .config import ExperimentConfig
from .metrics import multilabel_metrics, select_f1_thresholds
from .models import AttentionInstrumentModel, BaselineInstrumentModel, MaxInstrumentModel

try:
    import torch
except ImportError as exc:
    raise RuntimeError(
        "Training requires the optional dependency: python -m pip install -e '.[ml]'"
    ) from exc


MODEL_TYPES = {
    "baseline": BaselineInstrumentModel,
    "max": MaxInstrumentModel,
    "attention": AttentionInstrumentModel,
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(config: ExperimentConfig):
    model_class = MODEL_TYPES[config.model_type]
    return model_class(
        n_mels=config.n_mels,
        num_instruments=len(config.target_instruments),
        hidden_size=config.hidden_size,
    )


def validate_arrays(
    features: np.ndarray, labels: np.ndarray, config: ExperimentConfig
) -> Tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError("features must be shaped [clips, mel, time]")
    if labels.ndim != 2 or len(labels) != len(features):
        raise ValueError("labels must be shaped [clips, classes]")
    if features.shape[1] != config.n_mels:
        raise ValueError("feature mel dimension does not match config")
    if labels.shape[1] != len(config.target_instruments):
        raise ValueError("label class dimension does not match config")
    if not np.isfinite(features).all() or not np.isfinite(labels).all():
        raise ValueError("features and labels must be finite")
    if np.any((labels < 0) | (labels > 1)):
        raise ValueError("labels must be in [0, 1]")
    return features, labels


def predict_probabilities(model, features, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    tensor = torch.as_tensor(features, dtype=torch.float32)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size].to(device)
            outputs.append(model(batch)["clip_probabilities"].cpu().numpy())
    return np.concatenate(outputs, axis=0)


def train_arrays(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    config: ExperimentConfig,
    output_dir: Path,
    class_names: Optional[Sequence[str]] = None,
    patience: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """Train one model and tune decision thresholds on validation data only."""

    train_features, train_labels = validate_arrays(train_features, train_labels, config)
    validation_features, validation_labels = validate_arrays(
        validation_features, validation_labels, config
    )
    if not len(train_features) or not len(validation_features):
        raise ValueError("training and validation sets must be non-empty")
    patience = config.early_stopping_patience if patience is None else patience
    if patience < 1:
        raise ValueError("patience must be positive")
    names = list(class_names or config.target_instruments)
    if names != list(config.target_instruments):
        raise ValueError("class names/order must exactly match target_instruments")

    seed_everything(config.seed)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = build_model(config).to(selected_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    x_train = torch.from_numpy(train_features)
    y_train = torch.from_numpy(train_labels)
    generator = torch.Generator().manual_seed(config.seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best = None
    stale_epochs = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        permutation = torch.randperm(len(x_train), generator=generator)
        loss_sum = 0.0
        for start in range(0, len(x_train), config.batch_size):
            indices = permutation[start : start + config.batch_size]
            batch_x = x_train[indices].to(selected_device)
            batch_y = y_train[indices].to(selected_device)
            output = model(batch_x)
            loss = criterion(output["clip_logits"], batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(indices)

        probabilities = predict_probabilities(
            model, validation_features, config.batch_size, selected_device
        )
        thresholds = select_f1_thresholds(validation_labels, probabilities)
        metrics = multilabel_metrics(validation_labels, probabilities, thresholds, names)
        macro_f1 = float(metrics["macro"]["f1"])
        history.append({
            "epoch": epoch,
            "training_bce": loss_sum / len(x_train),
            "validation_macro_f1": macro_f1,
            "validation_micro_f1": float(metrics["micro"]["f1"]),
        })
        candidate = (macro_f1, -history[-1]["training_bce"])
        if best is None or candidate > best["selection"]:
            best = {
                "selection": candidate,
                "epoch": epoch,
                "state_dict": {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                },
                "thresholds": thresholds.copy(),
                "metrics": metrics,
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    checkpoint = {
        "format_version": 1,
        "config": config.to_dict(),
        "class_names": names,
        "best_epoch": best["epoch"],
        "validation_thresholds": best["thresholds"].tolist(),
        "validation_metrics": best["metrics"],
        "model_state_dict": best["state_dict"],
    }
    checkpoint_path = output_dir / "best_checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "model_type": config.model_type,
        "device": selected_device,
        "seed": config.seed,
        "epochs_completed": len(history),
        "best_epoch": best["epoch"],
        "validation_thresholds": best["thresholds"].tolist(),
        "validation_metrics": best["metrics"],
        "checkpoint": str(checkpoint_path),
    }
    (output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def load_checkpoint(path: Path, device: str = "cpu"):
    # Checkpoints are produced locally by this project and contain config plus
    # tensors, so the full dictionary format is intentionally loaded.
    checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    config = ExperimentConfig(**checkpoint["config"]).validate()
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config, checkpoint


def evaluate_checkpoint(
    checkpoint_path: Path,
    features: np.ndarray,
    labels: np.ndarray,
    device: str = "cpu",
) -> Dict[str, object]:
    """Evaluate with thresholds frozen in the selected validation checkpoint."""

    model, config, checkpoint = load_checkpoint(checkpoint_path, device)
    features, labels = validate_arrays(features, labels, config)
    probabilities = predict_probabilities(model, features, config.batch_size, device)
    return multilabel_metrics(
        labels,
        probabilities,
        checkpoint["validation_thresholds"],
        checkpoint["class_names"],
    )


def load_npz_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(str(path), allow_pickle=False) as archive:
        if "features" not in archive or "labels" not in archive:
            raise ValueError("NPZ dataset must contain features and labels arrays")
        return archive["features"], archive["labels"]
