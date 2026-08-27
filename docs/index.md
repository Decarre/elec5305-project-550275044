# Attention-Based Temporal Localization of Musical Instruments

**Author: Decarre**

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

## References

1. E. J. Humphrey, S. Durand, and B. McFee, "OpenMIC-2018: An Open Dataset for Multiple Instrument Recognition," *Proceedings of ISMIR*, 2018. [Paper](https://archives.ismir.net/ismir2018/paper/000248.pdf)
2. S. Gururani, M. Sharma, and A. Lerch, "An Attention Mechanism for Musical Instrument Recognition," *Proceedings of ISMIR*, 2019. [Paper](https://archives.ismir.net/ismir2019/paper/000007.pdf)
3. R. Bittner, J. Salamon, M. Tierney, M. Mauch, C. Cannam, and J. P. Bello, "MedleyDB: A Multitrack Dataset for Annotation-Intensive MIR Research," *Proceedings of ISMIR*, 2014. [Dataset website](https://medleydb.weebly.com/)

