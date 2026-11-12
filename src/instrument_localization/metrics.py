"""Dependency-light clip, frame and attention evaluation utilities."""

from typing import Dict, Optional, Sequence, Union

import numpy as np


ArrayLikeThreshold = Union[float, Sequence[float], np.ndarray]


def _as_samples(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim < 2:
        raise ValueError("values must have a final class dimension")
    return array.reshape(-1, array.shape[-1])


def _threshold_vector(thresholds: ArrayLikeThreshold, classes: int) -> np.ndarray:
    values = np.asarray(thresholds, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(classes, float(values))
    if values.shape != (classes,):
        raise ValueError("threshold count must match the class dimension")
    if np.any((values <= 0) | (values >= 1)):
        raise ValueError("thresholds must be between 0 and 1")
    return values


def multilabel_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: ArrayLikeThreshold = 0.5,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Compute per-class, macro and micro precision/recall/F1."""

    truth = _as_samples(np.asarray(targets) >= 0.5)
    scores = _as_samples(np.asarray(probabilities, dtype=np.float64))
    if truth.shape != scores.shape:
        raise ValueError("targets and probabilities must have matching shapes")
    threshold_values = _threshold_vector(thresholds, truth.shape[1])
    predicted = scores >= threshold_values[None, :]
    names = list(class_names or ["class_" + str(i) for i in range(truth.shape[1])])
    if len(names) != truth.shape[1]:
        raise ValueError("class name count must match the class dimension")

    per_class = []
    totals = CounterLike()
    for index, name in enumerate(names):
        tp = int(np.sum(predicted[:, index] & truth[:, index]))
        fp = int(np.sum(predicted[:, index] & ~truth[:, index]))
        fn = int(np.sum(~predicted[:, index] & truth[:, index]))
        tn = int(np.sum(~predicted[:, index] & ~truth[:, index]))
        totals.add(tp, fp, fn)
        row = _row(name, tp, fp, fn, threshold_values[index])
        row.update(negative_metrics(tp, fp, fn, tn))
        per_class.append(row)
    micro = _row("micro", totals.tp, totals.fp, totals.fn, None)
    macro = {
        metric: float(np.mean([row[metric] for row in per_class]))
        for metric in ("precision", "recall", "f1")
    }
    micro.update(negative_metrics(totals.tp, totals.fp, totals.fn, sum(r['tn'] for r in per_class)))
    for key in ('specificity', 'false_positive_rate', 'negative_f1', 'balanced_accuracy'):
        defined = [r[key] for r in per_class if r[key] is not None]
        macro[key] = float(np.mean(defined)) if defined else None
    return {
        "samples": int(truth.shape[0]),
        "per_class": per_class,
        "micro": micro,
        "macro": macro,
    }


def negative_metrics(tp, fp, fn, tn):
    """Undefined rates stay None; balanced accuracy requires both classes."""
    specificity = tn / (tn + fp) if tn + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        'tn': tn,
        'specificity': specificity,
        'false_positive_rate': fp / (fp + tn) if fp + tn else None,
        'negative_f1': 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0,
        'balanced_accuracy': (recall + specificity) / 2 if recall is not None and specificity is not None else None,
    }


def select_balanced_thresholds(targets, probabilities):
    """Validation-only threshold selection, with an explicit missing-class fallback."""
    truth, scores = _as_samples(targets), _as_samples(probabilities)
    selected = []
    for c in range(truth.shape[1]):
        if len(np.unique(truth[:, c])) < 2:
            selected.append(0.5)
            continue
        candidates = []
        for threshold in np.linspace(0.05, 0.95, 19):
            value = multilabel_metrics(truth[:, c:c+1], scores[:, c:c+1], threshold)['micro']['balanced_accuracy']
            candidates.append((value, -abs(threshold - 0.5), -threshold, threshold))
        selected.append(max(candidates)[-1])
    return np.asarray(selected)


class CounterLike:
    def __init__(self) -> None:
        self.tp = self.fp = self.fn = 0

    def add(self, tp: int, fp: int, fn: int) -> None:
        self.tp += tp
        self.fp += fp
        self.fn += fn


def _row(name: str, tp: int, fp: int, fn: int, threshold) -> Dict[str, object]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    row = {
        "class": name,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if threshold is not None:
        row["threshold"] = float(threshold)
    return row


def select_f1_thresholds(
    targets: np.ndarray,
    probabilities: np.ndarray,
    candidates: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Select one threshold per class using validation data only."""

    truth = _as_samples(np.asarray(targets) >= 0.5)
    scores = _as_samples(np.asarray(probabilities, dtype=np.float64))
    if truth.shape != scores.shape:
        raise ValueError("targets and probabilities must have matching shapes")
    grid = np.asarray(candidates or np.linspace(0.05, 0.95, 19), dtype=np.float64)
    if grid.ndim != 1 or not len(grid) or np.any((grid <= 0) | (grid >= 1)):
        raise ValueError("threshold candidates must lie between 0 and 1")
    selected = np.empty(truth.shape[1], dtype=np.float64)
    for class_index in range(truth.shape[1]):
        best = None
        for threshold in grid:
            row = multilabel_metrics(
                truth[:, class_index : class_index + 1],
                scores[:, class_index : class_index + 1],
                float(threshold),
            )["per_class"][0]
            # Prefer the threshold closest to 0.5 when F1 ties.
            candidate = (row["f1"], -abs(float(threshold) - 0.5), -float(threshold))
            if best is None or candidate > best[0]:
                best = (candidate, float(threshold))
        selected[class_index] = best[1]
    return selected


def attention_alignment(
    reference_activity: np.ndarray,
    attention_weights: np.ndarray,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Measure attention mass on active frames without calling it probability."""

    reference = np.asarray(reference_activity) >= 0.5
    attention = np.asarray(attention_weights, dtype=np.float64)
    if reference.shape != attention.shape or reference.ndim != 3:
        raise ValueError("reference and attention must match [clips, time, classes]")
    sums = attention.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-5):
        raise ValueError("attention weights must sum to one over time")
    names = list(class_names or ["class_" + str(i) for i in range(reference.shape[2])])
    rows = []
    for index, name in enumerate(names):
        valid = reference[:, :, index].any(axis=1)
        mass = (attention[:, :, index] * reference[:, :, index]).sum(axis=1)
        active_fraction = reference[:, :, index].mean(axis=1)
        # A value of 1.0 is the mass expected from uniform attention. This is
        # a relative concentration diagnostic, not an activity probability.
        lift = np.divide(
            mass,
            active_fraction,
            out=np.full_like(mass, np.nan, dtype=np.float64),
            where=active_fraction > 0,
        )
        rows.append({
            "class": name,
            "positive_clips": int(valid.sum()),
            "mean_attention_mass_on_active_frames": (
                float(mass[valid].mean()) if valid.any() else None
            ),
            "mean_active_frame_fraction": (
                float(active_fraction[valid].mean()) if valid.any() else None
            ),
            "mean_attention_lift_over_uniform": (
                float(np.nanmean(lift[valid])) if valid.any() else None
            ),
        })
    valid_values = [row["mean_attention_mass_on_active_frames"] for row in rows]
    valid_values = [value for value in valid_values if value is not None]
    return {"per_class": rows, "macro_active_mass": float(np.mean(valid_values)) if valid_values else None}
