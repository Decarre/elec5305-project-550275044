"""Temporal smoothing and instrument activity interval extraction."""

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class ActivityInterval:
    instrument: str
    start_time: float
    end_time: float
    mean_probability: float


def moving_average(probabilities: np.ndarray, window_size: int) -> np.ndarray:
    """Apply an edge-padded moving average along the time axis."""

    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("probabilities must be shaped [time, instruments]")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if window_size == 1:
        return values.copy()

    radius = window_size // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    smoothed = np.empty_like(values)
    for frame_index in range(values.shape[0]):
        smoothed[frame_index] = padded[frame_index : frame_index + window_size].mean(
            axis=0
        )
    return smoothed


def probabilities_to_intervals(
    probabilities: np.ndarray,
    instrument_names: Sequence[str],
    frame_hop_seconds: float,
    threshold: float = 0.5,
    smoothing_frames: int = 1,
    minimum_duration: float = 0.0,
) -> List[ActivityInterval]:
    """Convert frame probabilities into continuous instrument activity intervals."""

    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("probabilities must be shaped [time, instruments]")
    if values.shape[1] != len(instrument_names):
        raise ValueError("instrument name count must match probability columns")
    if frame_hop_seconds <= 0:
        raise ValueError("frame_hop_seconds must be positive")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")

    smoothed = moving_average(values, smoothing_frames)
    active = smoothed >= threshold
    intervals: List[ActivityInterval] = []

    for class_index, instrument in enumerate(instrument_names):
        start = None
        for frame_index in range(active.shape[0] + 1):
            is_active = frame_index < active.shape[0] and active[frame_index, class_index]
            if is_active and start is None:
                start = frame_index
            elif not is_active and start is not None:
                end = frame_index
                duration = (end - start) * frame_hop_seconds
                if duration >= minimum_duration:
                    intervals.append(
                        ActivityInterval(
                            instrument=str(instrument),
                            start_time=start * frame_hop_seconds,
                            end_time=end * frame_hop_seconds,
                            mean_probability=float(smoothed[start:end, class_index].mean()),
                        )
                    )
                start = None

    return intervals

