"""Core interfaces for temporal musical-instrument localization."""

from .config import ExperimentConfig, load_config
from .postprocess import ActivityInterval, probabilities_to_intervals

__all__ = [
    "ActivityInterval",
    "ExperimentConfig",
    "load_config",
    "probabilities_to_intervals",
]

__version__ = "0.1.0"

