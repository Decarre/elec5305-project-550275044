# Dataset setup

Dataset audio is not stored in this repository.

MedleyDB is the primary controlled dataset because it provides mixtures, stems, instrument metadata, and time-aligned stem-derived activation confidence. OpenMIC-2018 may be used to reproduce the published clip-level recognition baseline.

The official freely downloadable MedleyDB sample has been used for data-alignment and end-to-end engineering checks. It has only two songs and is insufficient for a valid train/validation/test performance study. Full MedleyDB audio must be requested from the official source before the artist-disjoint experiment can run.

Expected local resources (paths are supplied to scripts and are not committed):

```text
path/to/medleydb/          Official marl/medleydb metadata and annotations checkout
path/to/MedleyDB_sample/   Separately downloaded public two-track sample
path/to/MedleyDB/          Full audio after access is granted
```

## Prepared fallback

If MedleyDB access is still pending at the project decision gate, the main
quantitative experiment switches to the duplicate-free `Slakh2100-redux`
release. Expected local resources are:

```text
path/to/Slakh2100-redux/
  train/TrackXXXXX/{mix.flac,metadata.yaml,MIDI/,stems/}
  validation/TrackXXXXX/{mix.flac,metadata.yaml,MIDI/,stems/}
  test/TrackXXXXX/{mix.flac,metadata.yaml,MIDI/,stems/}
path/to/slakh-utils/splits/redux.json
```

Start with the official tiny subset to validate the adapter. Do not use the
original Slakh split without checking duplicate MIDI files across partitions.
See [`DATASET_FALLBACK_PLAN.md`](../DATASET_FALLBACK_PLAN.md) for the activation
date, leakage barrier, temporal references and experiment checkpoints.

