"""Baseline and attention-based multi-label model definitions."""

from typing import Dict

try:
    import torch
    from torch import Tensor, nn
except ImportError as exc:
    raise RuntimeError(
        "Models require the optional dependency: python -m pip install -e '.[ml]'"
    ) from exc


class FrameEncoder(nn.Module):
    """Encode log-mel frames while preserving the time axis."""

    def __init__(self, n_mels: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(n_mels, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
        )

    def forward(self, spectrogram: Tensor) -> Tensor:
        if spectrogram.ndim != 3:
            raise ValueError("spectrogram must be shaped [batch, mel, time]")
        return self.network(spectrogram).transpose(1, 2)


class BaselineInstrumentModel(nn.Module):
    """Global-mean-pooling baseline for clip-level prediction."""

    def __init__(self, n_mels: int, num_instruments: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.encoder = FrameEncoder(n_mels, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_instruments)

    def forward(self, spectrogram: Tensor) -> Dict[str, Tensor]:
        frame_features = self.encoder(spectrogram)
        frame_logits = self.classifier(frame_features)
        clip_logits = frame_logits.mean(dim=1)
        return {"clip_logits": clip_logits, "frame_logits": frame_logits}


class AttentionInstrumentModel(nn.Module):
    """Instrument-specific temporal attention pooling model."""

    def __init__(self, n_mels: int, num_instruments: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.encoder = FrameEncoder(n_mels, hidden_size)
        self.frame_classifier = nn.Linear(hidden_size, num_instruments)
        self.attention = nn.Linear(hidden_size, num_instruments)

    def forward(self, spectrogram: Tensor) -> Dict[str, Tensor]:
        frame_features = self.encoder(spectrogram)
        frame_logits = self.frame_classifier(frame_features)
        attention_weights = torch.softmax(self.attention(frame_features), dim=1)
        clip_logits = (attention_weights * frame_logits).sum(dim=1)
        return {
            "clip_logits": clip_logits,
            "frame_logits": frame_logits,
            "attention_weights": attention_weights,
        }

