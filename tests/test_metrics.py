import numpy as np

from instrument_localization.metrics import (
    attention_alignment,
    multilabel_metrics,
    select_f1_thresholds,
)


def test_validation_thresholds_and_metrics():
    targets = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
    probabilities = np.array([[0.2, 0.8], [0.3, 0.7], [0.6, 0.4], [0.7, 0.3]])
    thresholds = select_f1_thresholds(targets, probabilities, [0.25, 0.5, 0.75])
    result = multilabel_metrics(targets, probabilities, thresholds, ["a", "b"])
    assert result["micro"]["f1"] == 1.0
    assert result["macro"]["f1"] == 1.0


def test_attention_alignment_reports_active_mass():
    reference = np.array([[[1], [1], [0], [0]]])
    attention = np.array([[[0.4], [0.3], [0.2], [0.1]]])
    result = attention_alignment(reference, attention, ["guitar"])
    assert np.isclose(result["per_class"][0]["mean_attention_mass_on_active_frames"], 0.7)
