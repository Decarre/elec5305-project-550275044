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
    """Controlled global-mean baseline operating on frame probabilities."""

    def __init__(self, n_mels: int, num_instruments: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.encoder = FrameEncoder(n_mels, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_instruments)

    def forward(self, spectrogram: Tensor) -> Dict[str, Tensor]:
        frame_features = self.encoder(spectrogram)
        frame_logits = self.classifier(frame_features)
        frame_probabilities = torch.sigmoid(frame_logits)
        clip_probabilities = frame_probabilities.mean(dim=1)
        clip_logits = torch.logit(clip_probabilities.clamp(1e-7, 1.0 - 1e-7))
        return {
            "clip_logits": clip_logits,
            "frame_logits": frame_logits,
            "frame_probabilities": frame_probabilities,
            "clip_probabilities": clip_probabilities,
        }


class MaxInstrumentModel(nn.Module):
    """Optional global-max baseline using the same encoder and classifier."""

    def __init__(self, n_mels: int, num_instruments: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.encoder = FrameEncoder(n_mels, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_instruments)

    def forward(self, spectrogram: Tensor) -> Dict[str, Tensor]:
        frame_features = self.encoder(spectrogram)
        frame_logits = self.classifier(frame_features)
        frame_probabilities = torch.sigmoid(frame_logits)
        clip_probabilities = frame_probabilities.max(dim=1).values
        clip_logits = torch.logit(clip_probabilities.clamp(1e-7, 1.0 - 1e-7))
        return {
            "clip_logits": clip_logits,
            "frame_logits": frame_logits,
            "frame_probabilities": frame_probabilities,
            "clip_probabilities": clip_probabilities,
        }


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
        # Match the published AttentionMIC aggregation: bounded class scores
        # weighted by non-negative, time-normalised per-class attention.
        # Attention is relative importance across this clip, not P(active|t).
        raw_attention = torch.sigmoid(self.attention(frame_features))
        attention_weights = raw_attention / raw_attention.sum(dim=1, keepdim=True).clamp_min(1e-7)
        frame_probabilities = torch.sigmoid(frame_logits)
        clip_probabilities = (attention_weights * frame_probabilities).sum(dim=1)
        clip_logits = torch.logit(clip_probabilities.clamp(1e-7, 1.0 - 1e-7))
        return {
            "clip_logits": clip_logits,
            "frame_logits": frame_logits,
            "frame_probabilities": frame_probabilities,
            "clip_probabilities": clip_probabilities,
            "attention_weights": attention_weights,
        }

