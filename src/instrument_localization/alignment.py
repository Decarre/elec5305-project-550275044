"""Map centred STFT frame timestamps to half-open physical time bins."""
import numpy as np


def aggregate_time_grid(values, sample_rate=16000, hop_length=320,
                        duration=5.0, grid_seconds=0.5, mass=False):
    values = np.asarray(values)
    times = np.arange(values.shape[1]) * hop_length / sample_rate
    # Centred librosa STFT includes a frame at duration; it is outside the clip.
    bins = int(round(duration / grid_seconds))
    output = []
    for b in range(bins):
        mask = (times >= b * grid_seconds) & (times < (b + 1) * grid_seconds)
        if not mask.any():
            raise ValueError('no feature frames in reference bin')
        part = values[:, mask, :]
        output.append(part.sum(axis=1) if mass else part.mean(axis=1))
    result = np.stack(output, axis=1)
    if mass:
        denominator = result.sum(axis=1, keepdims=True)
        result = result / np.maximum(denominator, 1e-12)
    return result
