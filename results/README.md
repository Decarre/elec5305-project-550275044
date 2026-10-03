# Results

This directory contains reproducible preliminary outputs:

- `metadata_audit/`: official v1/v2 metadata coverage, a draft artist-grouped split and class co-occurrence counts.
- `sample_audit/`: audio/annotation alignment checks for the two-track official public sample.
- `sample_pilot/`: a training-set overfit check for the mean and attention paths.

The `sample_pilot` metrics are engineering diagnostics. The same two songs were used for fitting and measurement, so they are not held-out performance results and cannot compare generalisation. Source audio and model checkpoints are excluded from Git.

