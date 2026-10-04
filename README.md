# Attention-Based Temporal Localization of Musical Instruments

**Author:** Ryan Hu
**Status:** Reproducible experiment framework, dataset audit and controlled two-song engineering pilot completed (5 October 2026)

## Student information

- **Full name:** Ryan Hu
- **Student ID (SID):** 550275044
- **GitHub username:** Decarre
- **GitHub repository:** <https://github.com/Decarre/elec5305-project-550275044>
- **GitHub Project Site:** <https://decarre.github.io/elec5305-project-550275044/>
- **Project Feedback Two PDF:** [ELEC5305_Project_Feedback_Two_550275044.pdf](output/pdf/ELEC5305_Project_Feedback_Two_550275044.pdf)
- **Full proposal PDF:** [ELEC5305_Project_Proposal_550275044.pdf](ELEC5305_Project_Proposal_550275044.pdf)

## Overview

This project investigates whether an attention-based multi-label audio classification model can identify musical instruments in polyphonic music and estimate their entry and exit times.

The planned system will accept a music recording and produce an instrument activity timeline showing when each target instrument is predicted to be active.

## Research question

> When trained using only clip-level instrument labels, how accurately do frame-level class scores and instrument-specific attention signals recover the true temporal activity of instruments in polyphonic music?

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
5. Evaluate frame-level class scores and time-normalised attention weights as separate localisation signals.
6. Select thresholds and post-processing using validation artists only, then compare with held-out reference activations.
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

Frame class scores will be treated as candidate activity estimates. Attention weights are relative contributions that sum to one over time for each class; they are not activity probabilities and will be evaluated separately.

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

## Progress to date

The repository now contains a reproducible implementation and preliminary evidence:

- A validated experiment configuration object.
- Audio loading and log-mel feature extraction entry points.
- A convolutional encoder with global-mean and instrument-specific attention pooling models.
- AttentionMIC-style aggregation with separate frame scores, attention weights and clip probabilities.
- Temporal smoothing and conversion from frame probabilities to activity intervals.
- A command-line interface for configuration validation, training and frozen-checkpoint evaluation.
- Leakage-safe track and clip manifests created only after an artist-grouped split.
- Validation-only threshold selection stored in the selected checkpoint and reused unchanged at test time.
- Unit tests for model semantics, manifests, metrics, training/checkpoint reload and temporal post-processing (**13/13 passing**).
- A metadata audit of the official MedleyDB v1/v2 lists: **196 tracks, 116 artists**, and **117 tracks** with matching v2 activation-confidence files.
- Audio/annotation alignment checks on both tracks in the official public sample; all eight expected instrument stems match annotation columns.
- A 230-clip, two-song training-set sanity check using controlled frame-probability pooling for both mean and attention models.

The two-song run is an engineering overfit check: it verifies that both weak-label training paths execute and reduce loss, but it has no held-out artists and makes no generalisation claim. Full tables, figures, commands and limitations are on the [GitHub Project Site](https://decarre.github.io/elec5305-project-550275044/).

The exact relationship between the published AttentionMIC aggregation and this
controlled implementation is documented in [the implementation map](docs/attentionmic_mapping.md).

## Current challenges and points for feedback

The project currently has three main challenges:

1. **Dataset coverage and class imbalance.** The audit shows that strings have much weaker activity-reference coverage than drums, bass, guitar and piano. The final target list and artist split will be revised after the full audio is available.
2. **Reliability of attention for temporal localisation.** Attention may highlight contextual or correlated sounds rather than the true activity of a target instrument. The attention curves will be compared with frame-level reference activations and with a global-pooling baseline instead of being treated as explanations by default.
3. **Sensitivity of temporal post-processing and evaluation.** Activity intervals depend on probability thresholds, smoothing, minimum-duration rules, and event-matching tolerances. These settings will be selected using validation data, and their effects will be reported through ablation and sensitivity analysis.

The next valid performance experiment requires the full MedleyDB audio. It will split artists before clipping, tune all thresholds on validation data, and reserve held-out artists for the reported clip and frame metrics.

### Quick start

Create a Python environment and install the project in editable mode:

```bash
python -m pip install -e ".[dev,audio,ml]"
```

Validate and display the default experiment configuration:

```bash
python -m instrument_localization --config configs/attention.yaml --dry-run
```

The same validation is available through the structured CLI:

```bash
python -m instrument_localization validate --config configs/attention.yaml
```

After leakage-safe manifests and feature arrays have been produced, train with
clip-level labels and select the best epoch/thresholds on validation data:

```bash
python -m instrument_localization train --config configs/attention.yaml --train-data path/to/train.npz --validation-data path/to/validation.npz --output-dir runs/attention
```

The NPZ files contain `features` shaped `[clips, mel, time]` and clip-level
`labels` shaped `[clips, classes]`. Temporal reference arrays are deliberately
absent from this training interface. Evaluate once with the thresholds frozen
inside the selected checkpoint:

```bash
python -m instrument_localization evaluate --checkpoint runs/attention/best_checkpoint.pt --data path/to/test.npz --output-json runs/attention/test_metrics.json
```

Run the initial tests:

```bash
pytest
```

Reproduce the metadata audit and public-sample checks after obtaining the official resources separately:

```bash
python scripts/audit_medleydb.py --reference-root path/to/medleydb --output-dir results/metadata_audit
python scripts/audit_medleydb_sample.py --sample-root path/to/MedleyDB_sample --output-dir results/sample_audit
python scripts/run_sample_pilot.py --sample-root path/to/MedleyDB_sample --output-dir results/sample_pilot --epochs 25 --seed 5305
```

Build track/clip manifests after the artist split, using `--engineering-sample`
only for the two-song public sample:

```bash
python scripts/build_manifests.py --tracks-csv results/metadata_audit/tracks.csv --audio-root path/to/audio --reference-root path/to/medleydb --output-dir path/to/manifests
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
scripts/    Reproducible dataset audits and preliminary experiments
```

Audio datasets, trained model files, and other large generated artifacts will not be committed directly to the repository.

## References

1. R. M. Bittner, J. Salamon, M. Tierney, M. Mauch, C. Cannam, and J. P. Bello, "MedleyDB: A multitrack dataset for annotation-intensive MIR research," in *Proc. 15th Int. Soc. Music Inf. Retrieval Conf. (ISMIR)*, Taipei, Taiwan, 2014, pp. 155-160. [Online]. Available: <https://doi.org/10.5281/zenodo.1417889>
2. E. J. Humphrey, S. Durand, and B. McFee, "OpenMIC-2018: An open dataset for multiple instrument recognition," in *Proc. 19th Int. Soc. Music Inf. Retrieval Conf. (ISMIR)*, Paris, France, 2018, pp. 438-444. [Online]. Available: <https://archives.ismir.net/ismir2018/paper/000248.pdf>
3. S. Gururani, M. Sharma, and A. Lerch, "An attention mechanism for musical instrument recognition," in *Proc. 20th Int. Soc. Music Inf. Retrieval Conf. (ISMIR)*, Delft, The Netherlands, 2019, pp. 83-90. [Online]. Available: <https://archives.ismir.net/ismir2019/paper/000007.pdf>
4. C. Wang, G. Richard, and B. McFee, "Transfer learning and bias correction with pre-trained audio embeddings," in *Proc. 24th Int. Soc. Music Inf. Retrieval Conf. (ISMIR)*, Milan, Italy, 2023, pp. 64-70. [Online]. Available: <https://archives.ismir.net/ismir2023/paper/000006.pdf>
5. L. Ou, Y. Takahashi, and Y. Wang, "Lead instrument detection from multitrack music," in *Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP)*, 2025, pp. 1-5, doi: 10.1109/ICASSP49660.2025.10889928. [Online]. Available: <https://doi.org/10.1109/ICASSP49660.2025.10889928>
6. A. Kumar and B. Raj, "Audio event detection using weakly labeled data," in *Proc. ACM Multimedia*, 2016. [Online]. Available: <https://arxiv.org/abs/1605.02401>
7. Q. Kong, Y. Xu, W. Wang, and M. D. Plumbley, "Audio Set classification with attention model: A probabilistic perspective," in *Proc. ICASSP*, 2018. [Online]. Available: <https://arxiv.org/abs/1711.00927>
8. S. Jain and B. C. Wallace, "Attention is not Explanation," in *Proc. NAACL-HLT*, 2019. [Online]. Available: <https://aclanthology.org/N19-1357/>
