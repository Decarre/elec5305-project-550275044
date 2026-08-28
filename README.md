# Attention-Based Temporal Localization of Musical Instruments

**Author:** Ryan Hu
**Status:** Project proposal and initial implementation framework

## Student information

- **Full name:** Ryan Hu
- **Student ID (SID):** 550275044
- **GitHub username:** Decarre
- **GitHub repository:** <https://github.com/Decarre/elec5305-project-550275044>
- **GitHub Project Site:** <https://decarre.github.io/elec5305-project-550275044/>
- **Project feedback submission PDF:** [ELEC5305_Project_Feedback_Submission_550275044.pdf](ELEC5305_Project_Feedback_Submission_550275044.pdf)
- **Full proposal PDF:** [ELEC5305_Project_Proposal_550275044.pdf](ELEC5305_Project_Proposal_550275044.pdf)

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

## Current challenges and points for feedback

The project currently has three main challenges:

1. **Dataset coverage and class imbalance.** Some instrument families may have too few independent recordings for reliable training and evaluation. The final set of approximately five target families will therefore be selected only after a class-frequency and recording-level audit.
2. **Reliability of attention for temporal localisation.** Attention may highlight contextual or correlated sounds rather than the true activity of a target instrument. The attention curves will be compared with frame-level reference activations and with a global-pooling baseline instead of being treated as explanations by default.
3. **Sensitivity of temporal post-processing and evaluation.** Activity intervals depend on probability thresholds, smoothing, minimum-duration rules, and event-matching tolerances. These settings will be selected using validation data, and their effects will be reported through ablation and sensitivity analysis.

These challenges will be addressed during dataset preparation, model development, and evaluation. Teaching staff feedback will be used to refine the project scope and methodology as the project progresses.

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

1. R. M. Bittner, J. Salamon, M. Tierney, M. Mauch, C. Cannam, and J. P. Bello, "MedleyDB: A multitrack dataset for annotation-intensive MIR research," in *Proc. 15th Int. Soc. Music Inf. Retrieval Conf. (ISMIR)*, Taipei, Taiwan, 2014, pp. 155-160. [Online]. Available: <https://doi.org/10.5281/zenodo.1417889>
2. E. J. Humphrey, S. Durand, and B. McFee, "OpenMIC-2018: An open dataset for multiple instrument recognition," in *Proc. 19th Int. Soc. Music Inf. Retrieval Conf. (ISMIR)*, Paris, France, 2018, pp. 438-444. [Online]. Available: <https://archives.ismir.net/ismir2018/paper/000248.pdf>
3. S. Gururani, M. Sharma, and A. Lerch, "An attention mechanism for musical instrument recognition," in *Proc. 20th Int. Soc. Music Inf. Retrieval Conf. (ISMIR)*, Delft, The Netherlands, 2019, pp. 83-90. [Online]. Available: <https://archives.ismir.net/ismir2019/paper/000007.pdf>
4. C. Wang, G. Richard, and B. McFee, "Transfer learning and bias correction with pre-trained audio embeddings," in *Proc. 24th Int. Soc. Music Inf. Retrieval Conf. (ISMIR)*, Milan, Italy, 2023, pp. 64-70. [Online]. Available: <https://archives.ismir.net/ismir2023/paper/000006.pdf>
5. L. Ou, Y. Takahashi, and Y. Wang, "Lead instrument detection from multitrack music," in *Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP)*, 2025, pp. 1-5, doi: 10.1109/ICASSP49660.2025.10889928. [Online]. Available: <https://doi.org/10.1109/ICASSP49660.2025.10889928>
