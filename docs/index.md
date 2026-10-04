# Weakly Supervised Temporal Localisation of Musical Instruments

**Ryan Hu · SID 550275044 · progress update: 5 October 2026**

[Repository](https://github.com/Decarre/elec5305-project-550275044) · [Project Feedback Two PDF](https://github.com/Decarre/elec5305-project-550275044/blob/main/output/pdf/ELEC5305_Project_Feedback_Two_550275044.pdf) · [original proposal PDF](https://github.com/Decarre/elec5305-project-550275044/blob/main/ELEC5305_Project_Proposal_550275044.pdf)

## Research question

> When trained using only clip-level instrument labels, how accurately do frame-level class scores and instrument-specific attention signals recover the true temporal activity of instruments in polyphonic music?

The controlled comparison will use one shared frame encoder and change only the temporal aggregation: global mean pooling versus instrument-specific attention pooling. Both clip-level recognition and frame-level localisation will be evaluated.

This wording reflects an important distinction raised in feedback. A frame class score is a candidate estimate of activity at one time. An attention weight is a relative contribution to the clip decision and is normalised across time for each class. It is therefore not an activity probability. The project will evaluate frame scores, attention weights, and optionally their product as separate signals.

## System design

1. Audit MedleyDB metadata and select well-supported instrument families.
2. Assign complete artists to train, validation, or test before making short clips.
3. Extract 64-bin log-mel spectrograms from the audio.
4. Train mean-pooling and AttentionMIC-style models using only clip-level multi-label targets.
5. Choose class thresholds and temporal post-processing on validation data only.
6. Freeze those choices, then compare clip-level recognition and frame-level localisation on held-out artists.
7. Relate errors to class co-occurrence and inspect representative spectrogram/audio examples.

MedleyDB stem activation confidence is reserved for temporal evaluation. These annotations are automatically derived from isolated stems, so they are useful references rather than perfect manual frame labels. Frame precision, recall and F1 at a stated time grid will be primary; event and onset/offset results will be secondary.

## How the first feedback was incorporated

| Feedback direction | Implemented evidence |
|---|---|
| Start from published AttentionMIC code | Published aggregation and local modifications are separated in the [implementation map](attentionmic_mapping.html). |
| Do not treat attention as activity probability | Frame probabilities, attention weights and weighted contributions are exposed and evaluated as distinct signals. |
| Use an artist-level split before clipping | Deterministic artist assignment and leakage checks run before track windows are added to the clip manifest. |
| Keep pooling comparisons controlled | Mean, max and attention models share the frame encoder and aggregate the same bounded frame probabilities. |
| Select post-processing on validation data | Per-class validation thresholds are saved in the best checkpoint and reused unchanged by evaluation. |
| Keep temporal labels hidden during training | The training NPZ interface accepts only spectrogram features and clip-level labels. |

## Work completed to date

### Published-code study and model implementation

The project now uses the aggregation structure in the published [AttentionMIC implementation](https://github.com/SiddGururani/AttentionMIC): bounded frame class scores are combined with non-negative, class-specific attention weights normalised over time. The implementation exposes:

- `frame_probabilities[t, c]`: candidate frame localisation scores;
- `attention_weights[t, c]`: relative weights satisfying `sum_t weight[t,c] = 1`;
- `clip_probabilities[c]`: the attention-weighted sum of frame scores.

The mean and attention models share the same convolutional frame encoder. The
mean baseline averages the same bounded frame probabilities that the attention
model weights, so the comparison changes only temporal aggregation. A detailed
[AttentionMIC implementation map](attentionmic_mapping.html) separates the
published aggregation rule from local architectural choices. Automated tests
cover model semantics, manifests, metrics, training/checkpoint reload and
temporal post-processing; all **13 tests pass** in the current environment.

### Reproducible experiment framework

The repository now contains deterministic, class-aware artist splits and
track/clip manifests with stable hashes. Clips are created after splitting, and
the training NPZ interface contains log-mel features and clip labels only. The
time-aligned activity references stay outside the training interface.

The command-line workflow validates configuration, trains a selected pooling
model, chooses class thresholds on validation data, saves those thresholds in
the best checkpoint, and evaluates later data with the stored values unchanged.
This makes the planned held-out experiment auditable when full MedleyDB access
becomes available.

### MedleyDB metadata audit

The audit used the official MedleyDB v1/v2 track lists and metadata at `marl/medleydb` commit `537bb7b`. It found **196 tracks from 116 artists**. The checked-out annotation release contains version-2 activation-confidence files for **117 of these tracks**.

| Candidate family | Tracks | Artists | Tracks with v2 activity reference | Artists with v2 activity reference |
|---|---:|---:|---:|---:|
| drums | 123 | 87 | 94 | 74 |
| bass | 122 | 91 | 90 | 74 |
| guitar | 93 | 70 | 68 | 59 |
| piano | 86 | 51 | 46 | 34 |
| strings | 55 | 30 | 23 | 17 |

A deterministic artist-grouped draft split contains 136/33/27 tracks from 85/11/20 artists for train/validation/test. It is a planning result based on metadata and will be rechecked against downloaded audio and class balance before training. Strings are currently the weakest candidate: the draft split has only one validation artist and two test artists with an activity reference. The final class list or split therefore needs adjustment.

The audit also confirms strong co-occurrence: for example, drums and bass appear together in 105 tracks. This supports the planned analysis of whether a model uses a correlated instrument as evidence for the requested class.

[Metadata audit script](https://github.com/Decarre/elec5305-project-550275044/blob/main/scripts/audit_medleydb.py) · [class coverage CSV](https://github.com/Decarre/elec5305-project-550275044/blob/main/results/metadata_audit/class_coverage.csv) · [co-occurrence CSV](https://github.com/Decarre/elec5305-project-550275044/blob/main/results/metadata_audit/class_cooccurrence.csv)

### Official sample audio and annotation QA

The freely downloadable MedleyDB sample contains two songs from two artists. Both mixes are stereo at 44.1 kHz. The audit matched all **eight expected non-main-system stems** to their activation columns. Annotation frames are spaced by approximately **46.4 ms**, and each annotation ends within one frame of its mix.

| Track | Mix duration | Metadata stems | Activity columns | Alignment result |
|---|---:|---:|---:|---|
| LizNelson_Rainfall | 284.91 s | 5 | 5 | matched |
| Phoenix_ScotchMorris | 177.14 s | 4 (3 expected) | 3 | matched |

The Phoenix metadata also includes a `Main System` stem, which is intentionally absent from the instrument-activity columns; the other three columns match acoustic guitar, flute and violin.

![First 60 seconds of an official sample mix with stem-derived reference confidence](https://raw.githubusercontent.com/Decarre/elec5305-project-550275044/main/results/sample_audit/sample_alignment.png)

[Sample QA script](https://github.com/Decarre/elec5305-project-550275044/blob/main/scripts/audit_medleydb_sample.py) · [track summary CSV](https://github.com/Decarre/elec5305-project-550275044/blob/main/results/sample_audit/tracks.csv) · [stem summary CSV](https://github.com/Decarre/elec5305-project-550275044/blob/main/results/sample_audit/stems.csv)

### Two-song training sanity check

An end-to-end engineering check was run on 230 non-overlapping two-second clips from the two official sample songs. Clip labels were produced from the withheld time series for preprocessing only: a class was marked present when its activation confidence was at least 0.5 for at least 10% of the clip. Both models received only the resulting clip-level vectors during training.

| Model | Initial training BCE | Final training BCE | Training micro-F1 | Training macro-F1 |
|---|---:|---:|---:|---:|
| mean pooling | 0.4715 | 0.0357 | 0.8721 | 0.7717 |
| attention pooling | 0.4509 | 0.0135 | 0.9977 | 0.9967 |

These are **training-set overfit diagnostics**. Training and evaluation used the same two songs, guitar is positive in every clip, and only two artists are available. The mean result changed from the earlier pilot because the corrected controlled baseline averages frame probabilities rather than applying a sigmoid after averaging logits. The values demonstrate that feature extraction, weak-label training and both aggregation paths run end to end; they do not estimate generalisation and cannot establish that one pooling method is better.

The figure below illustrates why the temporal signals must be separated. The reference confidence, each model's frame score and the attention weight are different quantities. Although the model was given only a positive clip label, its frame score and attention may emphasize different parts of the clip.

![Reference confidence, model frame scores and relative attention weight](https://raw.githubusercontent.com/Decarre/elec5305-project-550275044/main/results/sample_pilot/pilot_signals.png)

[Sanity-check script](https://github.com/Decarre/elec5305-project-550275044/blob/main/scripts/run_sample_pilot.py) · [saved run summary](https://github.com/Decarre/elec5305-project-550275044/blob/main/results/sample_pilot/summary.json) · [per-class training metrics](https://github.com/Decarre/elec5305-project-550275044/blob/main/results/sample_pilot/training_metrics.csv)

## Reproduction

Install the project and optional audio/model dependencies in an isolated environment:

```bash
python -m pip install -e ".[dev,audio,ml]"
python -m pytest
```

With a checkout of the official metadata tools and the separately downloaded official sample:

```bash
python scripts/audit_medleydb.py --reference-root path/to/medleydb --output-dir results/metadata_audit
python scripts/audit_medleydb_sample.py --sample-root path/to/MedleyDB_sample --output-dir results/sample_audit
python scripts/run_sample_pilot.py --sample-root path/to/MedleyDB_sample --output-dir results/sample_pilot --epochs 25 --seed 5305
```

The audio is excluded from Git. The repository contains scripts and derived summaries only.

## Next experiment

The next valid performance experiment requires the full MedleyDB audio. After access is available I will:

1. verify every chosen track, duration and activity-reference file;
2. revise the artist split so all selected families have credible validation/test support;
3. create clips only after the artist split;
4. train matched mean and attention models;
5. select thresholds/smoothing on validation data and freeze them;
6. report held-out clip and frame precision, recall and F1;
7. evaluate frame scores and attention separately, then analyse co-occurrence and failure cases.

OpenMIC remains useful for reproducing the published AttentionMIC recognition baseline, but cross-dataset transfer is an extension because it adds domain mismatch. A strongly supervised frame model is also an optional upper bound rather than the main weak-supervision experiment.

## References and software resources

1. R. M. Bittner et al., “MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research,” ISMIR, 2014. [Paper and dataset](https://medleydb.weebly.com/)
2. E. J. Humphrey, S. Durand, and B. McFee, “OpenMIC-2018: An Open Dataset for Multiple Instrument Recognition,” ISMIR, 2018. [Paper](https://archives.ismir.net/ismir2018/paper/000248.pdf)
3. S. Gururani, M. Sharma, and A. Lerch, “An Attention Mechanism for Musical Instrument Recognition,” ISMIR, 2019. [Paper](https://archives.ismir.net/ismir2019/paper/000007.pdf)
4. A. Kumar and B. Raj, “Audio Event Detection using Weakly Labeled Data,” ACM Multimedia, 2016. [Paper](https://arxiv.org/abs/1605.02401)
5. Q. Kong, Y. Xu, W. Wang, and M. D. Plumbley, “Audio Set Classification with Attention Model: A Probabilistic Perspective,” ICASSP, 2018. [Paper](https://arxiv.org/abs/1711.00927)
6. S. Jain and B. C. Wallace, “Attention is not Explanation,” NAACL, 2019. [Paper](https://aclanthology.org/N19-1357/)
7. C. Wang, G. Richard, and B. McFee, “Transfer Learning and Bias Correction with Pre-trained Audio Embeddings,” ISMIR, 2023. [Paper](https://archives.ismir.net/ismir2023/paper/000006.pdf)
8. L. Ou, Y. Takahashi, and Y. Wang, “Lead Instrument Detection from Multitrack Music,” ICASSP, 2025. [DOI](https://doi.org/10.1109/ICASSP49660.2025.10889928)

Software used as references: [AttentionMIC](https://github.com/SiddGururani/AttentionMIC), [OpenMIC-2018 tools](https://github.com/cosmir/openmic-2018), and [MedleyDB annotations/tools](https://github.com/marl/medleydb).
