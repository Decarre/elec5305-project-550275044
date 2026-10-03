"""MedleyDB activation-confidence parsing and time-grid alignment."""

from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


def read_activation_confidence(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    values = np.genfromtxt(str(path), delimiter=",", names=True, dtype=np.float32)
    if values.dtype.names is None or "time" not in values.dtype.names:
        raise ValueError("activation file must contain a time column")
    times = np.atleast_1d(values["time"]).astype(np.float64)
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError("activation times must be strictly increasing")
    curves = {
        name: np.atleast_1d(values[name]).astype(np.float32)
        for name in values.dtype.names
        if name != "time"
    }
    return times, curves


def aggregate_stems_to_families(
    stem_curves: Mapping[str, np.ndarray],
    stem_to_family: Mapping[str, str],
    families: Sequence[str],
) -> np.ndarray:
    """Aggregate same-family stems with a frame-wise maximum."""

    lengths = {len(np.asarray(curve)) for curve in stem_curves.values()}
    if len(lengths) != 1:
        raise ValueError("all stem activation curves must have equal length")
    frames = next(iter(lengths), 0)
    result = np.zeros((frames, len(families)), dtype=np.float32)
    family_index = {family: index for index, family in enumerate(families)}
    for stem_id, curve in stem_curves.items():
        family = stem_to_family.get(stem_id)
        if family in family_index:
            column = family_index[family]
            result[:, column] = np.maximum(result[:, column], np.asarray(curve))
    return result


def resample_activity_grid(
    times: np.ndarray,
    activity: np.ndarray,
    start_time: float,
    end_time: float,
    grid_seconds: float,
    reducer: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate irregular/fine reference frames onto a reporting grid."""

    times = np.asarray(times, dtype=np.float64)
    activity = np.asarray(activity, dtype=np.float32)
    if activity.ndim != 2 or len(times) != len(activity):
        raise ValueError("activity must be shaped [time, classes]")
    if grid_seconds <= 0 or end_time <= start_time:
        raise ValueError("invalid grid interval")
    edges = np.arange(start_time, end_time + grid_seconds, grid_seconds)
    if edges[-1] < end_time:
        edges = np.append(edges, end_time)
    centres = edges[:-1] + np.diff(edges) / 2
    output = np.zeros((len(centres), activity.shape[1]), dtype=np.float32)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (times >= left) & (times < right)
        if mask.any():
            output[index] = activity[mask].max(axis=0) if reducer == "max" else activity[mask].mean(axis=0)
    if reducer not in {"mean", "max"}:
        raise ValueError("reducer must be 'mean' or 'max'")
    return centres, output
