"""Audio loading and log-mel feature extraction."""

from pathlib import Path
from typing import Tuple

import numpy as np


def _require_librosa():
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError(
            "Audio features require the optional dependency: "
            "python -m pip install -e '.[audio]'"
        ) from exc
    return librosa


def load_audio(path: Path, sample_rate: int) -> Tuple[np.ndarray, int]:
    """Load a mono audio file at the requested sample rate."""

    librosa = _require_librosa()
    waveform, actual_rate = librosa.load(str(path), sr=sample_rate, mono=True)
    return np.asarray(waveform, dtype=np.float32), int(actual_rate)


def log_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    """Convert a waveform to a log-mel spectrogram shaped [mel, time]."""

    librosa = _require_librosa()
    mel_power = librosa.feature.melspectrogram(
        y=np.asarray(waveform, dtype=np.float32),
        sr=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    return np.asarray(librosa.power_to_db(mel_power, ref=np.max), dtype=np.float32)

