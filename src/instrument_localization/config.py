"""Experiment configuration and validation."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class ExperimentConfig:
    """Settings shared by preprocessing, modelling, and evaluation."""

    sample_rate: int = 22050
    n_mels: int = 64
    n_fft: int = 2048
    hop_length: int = 512
    clip_duration: float = 10.0
    analysis_hop: float = 2.5
    target_instruments: List[str] = field(
        default_factory=lambda: ["drums", "bass", "guitar", "piano", "strings"]
    )
    decision_threshold: float = 0.5
    smoothing_frames: int = 5
    minimum_event_duration: float = 1.0
    model_type: str = "attention"
    hidden_size: int = 128
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    seed: int = 5305
    reporting_grid_seconds: float = 0.5
    early_stopping_patience: int = 8

    def validate(self) -> "ExperimentConfig":
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.n_mels <= 0 or self.n_fft <= 0 or self.hop_length <= 0:
            raise ValueError("spectrogram dimensions must be positive")
        if self.clip_duration <= 0 or self.analysis_hop <= 0:
            raise ValueError("clip durations and hops must be positive")
        if not self.target_instruments:
            raise ValueError("at least one target instrument is required")
        if len(set(self.target_instruments)) != len(self.target_instruments):
            raise ValueError("target instrument names must be unique")
        if not 0.0 < self.decision_threshold < 1.0:
            raise ValueError("decision_threshold must be between 0 and 1")
        if self.smoothing_frames < 1 or self.smoothing_frames % 2 == 0:
            raise ValueError("smoothing_frames must be a positive odd integer")
        if self.minimum_event_duration < 0:
            raise ValueError("minimum_event_duration cannot be negative")
        if self.model_type not in {"baseline", "max", "attention"}:
            raise ValueError("model_type must be 'baseline', 'max' or 'attention'")
        if self.hidden_size <= 0 or self.batch_size <= 0 or self.epochs <= 0:
            raise ValueError("training dimensions must be positive")
        if self.early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if self.learning_rate <= 0 or self.reporting_grid_seconds <= 0:
            raise ValueError("learning rate and reporting grid must be positive")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: Path) -> ExperimentConfig:
    """Load and validate an experiment configuration from YAML."""

    with Path(path).open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError("configuration must contain a YAML mapping")
    return ExperimentConfig(**values).validate()

