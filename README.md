# Attention-Based Temporal Localization of Musical Instruments

**Author:** Ryan Hu
**Status:** Project proposal and initial implementation framework

## Student information

- **Full name:** Ryan Hu
- **Student ID (SID):** 550275044
- **GitHub username:** Decarre
- **GitHub repository:** <https://github.com/Decarre/elec5305-project-550275044>
- **GitHub Project Site:** <https://decarre.github.io/elec5305-project-550275044/>
- **Project proposal PDF:** [ELEC5305_Project_Proposal_550275044.pdf](ELEC5305_Project_Proposal_550275044.pdf)

## Overview

This project investigates whether an attention-based multi-label audio classification model can identify musical instruments in polyphonic music and estimate their entry and exit times.

The planned system will accept a music recording and produce an instrument activity timeline showing when each target instrument is predicted to be active.

## Research question

> Can an attention-based multi-label model localize musical instrument activity and estimate instrument entry and exit times when trained using only clip-level instrument labels?

The project also investigates two supporting questions:

1. Does attention-based pooling improve instrument recognition compared with global mean or maximum pooling?
2. Which instruments are most frequently confused, and can these errors be related to similar timbre, low source energy, or instrument co-occurrence?

## Planned scope

- Polyphonic music rather than isolated instrument recordings.
- Approximately five common instrument families, selected after checking dataset coverage.
- Multi-label prediction because several instruments can be active simultaneously.
- Clip-level labels for model training.
- Time-aligned instrument activity annotations reserved for temporal evaluation.
- Song-level or artist-level data splits to prevent segments from the same recording appearing in both training and testing data.

## Proposed method

1. Load and standardise the audio recordings.
2. Convert the audio into log-mel spectrograms.
3. Implement a baseline convolutional model with global temporal pooling.
4. Implement an instrument-specific attention model.
5. Convert frame-level scores or attention weights into activity intervals using thresholding and temporal smoothing.
6. Compare predicted activity intervals with reference instrument activations.
7. Analyse instrument-specific errors and recurring confusion patterns.

## Evaluation

The planned evaluation includes:

- Clip-level micro-F1 and macro-F1.
- Frame-level micro-F1 and macro-F1.
- Per-instrument precision, recall, and F1.
- Instrument onset and offset timing error.
- Event-level F1 under a defined temporal tolerance.
- Per-label binary confusion matrices.
- Pairwise error and instrument co-occurrence analysis.

Attention values will be treated as candidate localization signals and evaluated against time-aligned annotations. They will not be assumed to provide reliable temporal explanations without quantitative validation.

## Expected output

The final demonstration is intended to show:

- Predicted instruments for an input recording.
- Instrument-specific probability or attention curves.
- Estimated entry and exit times.
- A visual instrument activity timeline.
- Representative correct predictions and failure cases.
- A summary of instrument pairs that are difficult for the model to distinguish.

## Dataset plan

MedleyDB is the primary candidate because it provides polyphonic mixes, stems, instrument metadata, and time-aligned instrument activation annotations. OpenMIC-2018 may be used as an additional source for clip-level multi-label instrument recognition experiments.

The final target classes and dataset split will be selected only after measuring class frequency and checking that each class has sufficient independent recordings for training, validation, and testing.

## Initial implementation framework

The repository now contains a small implementation framework for the planned experiments:

- A validated experiment configuration object.
- Audio loading and log-mel feature extraction entry points.
- A convolutional encoder with global-mean and instrument-specific attention pooling models.
- Temporal smoothing and conversion from frame probabilities to activity intervals.
- A dry-run command-line interface.
- Unit tests for configuration, model output shapes, and temporal post-processing.

This framework defines the experiment interfaces but does not include a trained model or claim experimental results.

### Quick start

Create a Python environment and install the project in editable mode:

```bash
python -m pip install -e ".[dev,audio,ml]"
```

Validate and display the default experiment configuration:

```bash
python -m instrument_localization --config configs/attention.yaml --dry-run
```

Run the initial tests:

```bash
pytest
```

## Repository structure

```text
data/       Dataset instructions and metadata only
configs/    Reproducible experiment configurations
docs/       GitHub Project Site
notebooks/  Reproducible exploration and experiments
results/    Generated figures, tables, and example timelines
src/        Reusable preprocessing, modelling, and evaluation code
tests/      Unit tests for reusable project components
```

Audio datasets, trained model files, and other large generated artifacts will not be committed directly to the repository.

## References

1. E. J. Humphrey, S. Durand, and B. McFee, "OpenMIC-2018: An Open Dataset for Multiple Instrument Recognition," *Proceedings of ISMIR*, 2018. <https://archives.ismir.net/ismir2018/paper/000248.pdf>
2. S. Gururani, M. Sharma, and A. Lerch, "An Attention Mechanism for Musical Instrument Recognition," *Proceedings of ISMIR*, 2019. <https://archives.ismir.net/ismir2019/paper/000007.pdf>
3. R. Bittner, J. Salamon, M. Tierney, M. Mauch, C. Cannam, and J. P. Bello, "MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research," *Proceedings of ISMIR*, 2014. <https://medleydb.weebly.com/>
