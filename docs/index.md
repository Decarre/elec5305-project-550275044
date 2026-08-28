# Attention-Based Temporal Localization of Musical Instruments

**Author: Ryan Hu**

## Student information

- **Full name:** Ryan Hu
- **Student ID (SID):** 550275044
- **GitHub username:** Decarre
- **GitHub repository:** [elec5305-project-550275044](https://github.com/Decarre/elec5305-project-550275044)
- **Project feedback submission PDF:** [Brief Project Description](https://github.com/Decarre/elec5305-project-550275044/blob/main/ELEC5305_Project_Feedback_Submission_550275044.pdf)
- **Full proposal PDF:** [ELEC5305 Project Proposal](https://github.com/Decarre/elec5305-project-550275044/blob/main/ELEC5305_Project_Proposal_550275044.pdf)

## Project proposal

Musical instrument recognition becomes difficult when several instruments are active at the same time. A conventional classifier may identify the instruments present in an entire audio clip but does not necessarily explain when each instrument enters or exits.

This project investigates an attention-based multi-label model for recognizing and temporally localizing instruments in polyphonic music. The intended output is an instrument activity timeline that displays the predicted entry and exit times of each target instrument.

## Research question

> Can an attention-based multi-label model localize musical instrument activity and estimate instrument entry and exit times when trained using only clip-level instrument labels?

The project will also examine whether attention-based pooling improves upon a conventional global-pooling baseline and which instruments are most frequently confused by the model.

## Approach

Audio recordings will first be represented as log-mel spectrograms. A convolutional baseline using global temporal pooling will be implemented and compared with an instrument-specific attention model. Frame-level model outputs will be converted into activity intervals using a defined threshold and temporal smoothing.

Approximately five common instrument families will be selected after examining the available data. MedleyDB is the primary candidate dataset because it provides polyphonic recordings, individual stems, instrument metadata, and time-aligned instrument activity annotations. During the weakly supervised experiment, only clip-level labels will be supplied during training. The time-aligned annotations will be reserved for evaluation.

## Experiments and evaluation

The baseline and attention models will be compared using clip-level and frame-level precision, recall, micro-F1, macro-F1, and per-instrument F1. Temporal localization will be evaluated through instrument onset and offset errors and event-level F1 under a clearly defined timing tolerance.

The error analysis will include per-label binary confusion matrices, pairwise error patterns, instrument co-occurrence statistics, spectrograms, and representative audio examples. This analysis will investigate whether errors are associated with similar timbre, low source energy, or instruments that frequently occur together.

Attention weights will not automatically be treated as correct explanations. Their temporal locations will be compared quantitatively with the reference activation annotations.

## Planned demonstration

The final system is intended to accept a music recording and generate:

- Predicted instrument labels.
- Instrument-specific temporal probability or attention curves.
- Estimated entry and exit times.
- A visual instrument activity timeline.
- Representative successful and unsuccessful predictions.
- A summary of frequently confused instrument pairs.

## Current implementation status

An initial Python framework has been created for reproducible configuration, log-mel feature extraction, baseline temporal pooling, instrument-specific attention pooling, and conversion of frame probabilities into activity intervals. Unit tests cover the initial model interfaces and temporal post-processing. Model training and dataset experiments have not yet been completed.

## Current challenges and points for feedback

The project currently has three main challenges:

1. **Dataset coverage and class imbalance.** Some instrument families may have too few independent recordings for reliable training and evaluation. The final set of approximately five target families will therefore be selected only after a class-frequency and recording-level audit.
2. **Reliability of attention for temporal localisation.** Attention may highlight contextual or correlated sounds rather than the true activity of a target instrument. The attention curves will be compared with frame-level reference activations and with a global-pooling baseline instead of being treated as explanations by default.
3. **Sensitivity of temporal post-processing and evaluation.** Activity intervals depend on probability thresholds, smoothing, minimum-duration rules, and event-matching tolerances. These settings will be selected using validation data, and their effects will be reported through ablation and sensitivity analysis.

These challenges will be addressed during dataset preparation, model development, and evaluation. Teaching staff feedback will be used to refine the project scope and methodology as the project progresses.

## References

1. R. M. Bittner, J. Salamon, M. Tierney, M. Mauch, C. Cannam, and J. P. Bello, "MedleyDB: A multitrack dataset for annotation-intensive MIR research," in *Proc. 15th Int. Soc. Music Inf. Retrieval Conf. (ISMIR)*, Taipei, Taiwan, 2014, pp. 155-160. [Online]. Available: [https://doi.org/10.5281/zenodo.1417889](https://doi.org/10.5281/zenodo.1417889)
2. E. J. Humphrey, S. Durand, and B. McFee, "OpenMIC-2018: An open dataset for multiple instrument recognition," in *Proc. 19th Int. Soc. Music Inf. Retrieval Conf. (ISMIR)*, Paris, France, 2018, pp. 438-444. [Online]. Available: [Paper](https://archives.ismir.net/ismir2018/paper/000248.pdf)
3. S. Gururani, M. Sharma, and A. Lerch, "An attention mechanism for musical instrument recognition," in *Proc. 20th Int. Soc. Music Inf. Retrieval Conf. (ISMIR)*, Delft, The Netherlands, 2019, pp. 83-90. [Online]. Available: [Paper](https://archives.ismir.net/ismir2019/paper/000007.pdf)
4. C. Wang, G. Richard, and B. McFee, "Transfer learning and bias correction with pre-trained audio embeddings," in *Proc. 24th Int. Soc. Music Inf. Retrieval Conf. (ISMIR)*, Milan, Italy, 2023, pp. 64-70. [Online]. Available: [Paper](https://archives.ismir.net/ismir2023/paper/000006.pdf)
5. L. Ou, Y. Takahashi, and Y. Wang, "Lead instrument detection from multitrack music," in *Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP)*, 2025, pp. 1-5, doi: 10.1109/ICASSP49660.2025.10889928. [Online]. Available: [https://doi.org/10.1109/ICASSP49660.2025.10889928](https://doi.org/10.1109/ICASSP49660.2025.10889928)
