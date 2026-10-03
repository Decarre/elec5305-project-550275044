import json

import numpy as np

from instrument_localization.config import ExperimentConfig
from instrument_localization.training import (
    evaluate_checkpoint,
    load_checkpoint,
    predict_probabilities,
    train_arrays,
)


def test_training_checkpoint_freezes_validation_thresholds(tmp_path):
    rng = np.random.RandomState(5305)
    features = rng.normal(size=(18, 8, 12)).astype(np.float32)
    labels = np.stack([
        features[:, 0].mean(axis=1) > 0,
        features[:, 1].mean(axis=1) > 0,
    ], axis=1).astype(np.float32)
    config = ExperimentConfig(
        n_mels=8,
        target_instruments=["drums", "bass"],
        hidden_size=8,
        batch_size=4,
        epochs=3,
        learning_rate=0.01,
        model_type="baseline",
    ).validate()
    summary = train_arrays(
        features[:12], labels[:12], features[12:], labels[12:], config, tmp_path,
        patience=3, device="cpu",
    )
    checkpoint_path = tmp_path / "best_checkpoint.pt"
    assert checkpoint_path.is_file()
    assert (tmp_path / "history.csv").is_file()
    saved = json.loads((tmp_path / "validation_summary.json").read_text())
    assert saved["validation_thresholds"] == summary["validation_thresholds"]
    model, loaded_config, checkpoint = load_checkpoint(checkpoint_path)
    first = predict_probabilities(model, features[12:], loaded_config.batch_size, "cpu")
    model2, _, _ = load_checkpoint(checkpoint_path)
    second = predict_probabilities(model2, features[12:], loaded_config.batch_size, "cpu")
    assert np.allclose(first, second)
    result = evaluate_checkpoint(checkpoint_path, features[12:], labels[12:])
    assert result["samples"] == 6
    assert checkpoint["validation_thresholds"] == summary["validation_thresholds"]
