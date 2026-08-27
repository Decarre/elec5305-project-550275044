# Source code

The `instrument_localization` package contains the initial reusable framework:

- `config.py`: validated experiment settings and YAML loading.
- `features.py`: audio loading and log-mel spectrogram extraction.
- `models.py`: convolutional baseline and instrument-specific attention models.
- `postprocess.py`: smoothing and activity-interval extraction.
- `cli.py`: command-line configuration validation.

The package provides interfaces for the planned experiments. It does not include trained weights or completed experimental results.
