# ELEC5305 Project Feedback Submission

## Project Title

**Attention-Based Temporal Localisation of Musical Instruments in Polyphonic Music**

## Student Information

- **Full Name:** Ryan Hu
- **Student ID (SID):** 550275044
- **GitHub Project Site:** <https://decarre.github.io/elec5305-project-550275044/>
- **GitHub Repository:** <https://github.com/Decarre/elec5305-project-550275044>

## Project Description

This project investigates whether musical instruments can be identified and temporally localised in polyphonic music, where several instruments may be active at the same time. A conventional clip-level classifier can report that a recording contains guitar, bass, or drums, but it does not necessarily show when each instrument enters or exits. The proposed system will therefore produce multi-label instrument predictions together with an instrument activity timeline. The central research question is: **Can an attention-based model trained using clip-level instrument labels estimate instrument entry and exit times more effectively than a conventional global-pooling baseline?**

The project is motivated by the difficulty of recognising overlapping and masked instruments in ensemble recordings. MedleyDB provides multitrack recordings, stems, metadata, and time-aligned instrument activations that can support temporal evaluation [1]. OpenMIC-2018 provides a larger collection of polyphonic excerpts with multi-label instrument annotations [2]. Previous research suggests that instrument-specific attention can improve weakly supervised recognition and highlight relevant short-time regions [3]. However, attention weights will be treated as candidate localisation signals and tested against independent temporal annotations rather than assumed to be correct explanations.

## Proposed Project Work

The project will initially focus on approximately five sufficiently represented instrument families, selected after a class-frequency audit. Audio recordings will be divided by song into training, validation, and test sets to prevent segments from the same recording appearing in multiple splits. Each recording will be transformed into a log-mel spectrogram using a short-time Fourier transform. A convolutional model with global temporal pooling will be implemented as the baseline. The proposed model will use the same frame encoder but replace global pooling with instrument-specific temporal attention. Thresholding, temporal smoothing, and a minimum-duration rule will convert frame-level scores into predicted activity intervals.

MATLAB course examples may be used to verify STFT framing and construct a conventional feature baseline. Python, PyTorch, NumPy, and librosa will be used for neural modelling, temporal post-processing, evaluation, and visualisation. Recent work indicates that dataset and genre bias can influence instrument-recognition results [4], so the data split and class distribution will be reported clearly. The public GitHub Project Site will be updated with implementation progress, experimental results, figures, and changes made in response to teaching staff feedback.

## Evaluation and Expected Outcomes

The global-pooling and attention models will be compared using the same data splits. Clip-level evaluation will include micro-F1, macro-F1, precision, recall, and per-instrument F1. Temporal evaluation will include frame-level F1, event-level F1 under a stated tolerance, and mean onset and offset timing error. Error analysis will examine per-label confusion patterns and instrument co-occurrence to determine whether failures are associated with similar timbre, masking, class imbalance, or correlated arrangements. Recent frame-level attention research on lead-instrument detection provides a related comparison while addressing a different single-lead task [5].

The expected outcome is a reproducible prototype that accepts a compatible audio recording and generates instrument probabilities, temporal curves, estimated entry and exit times, and a visual activity timeline. This submission describes the proposed work and does not claim completed training results. Teaching staff feedback on the GitHub Project Site will be used to refine the project scope, modelling choices, and evaluation procedure before the final submission.

## References

1. R. M. Bittner, J. Salamon, M. Tierney, M. Mauch, C. Cannam, and J. P. Bello, "MedleyDB: A multitrack dataset for annotation-intensive MIR research," in *Proc. 15th Int. Soc. Music Inf. Retrieval Conf. (ISMIR)*, Taipei, Taiwan, 2014, pp. 155-160. [Online]. Available: <https://doi.org/10.5281/zenodo.1417889>
2. E. J. Humphrey, S. Durand, and B. McFee, "OpenMIC-2018: An open dataset for multiple instrument recognition," in *Proc. 19th Int. Soc. Music Inf. Retrieval Conf. (ISMIR)*, Paris, France, 2018, pp. 438-444. [Online]. Available: <https://archives.ismir.net/ismir2018/paper/000248.pdf>
3. S. Gururani, M. Sharma, and A. Lerch, "An attention mechanism for musical instrument recognition," in *Proc. 20th Int. Soc. Music Inf. Retrieval Conf. (ISMIR)*, Delft, The Netherlands, 2019, pp. 83-90. [Online]. Available: <https://archives.ismir.net/ismir2019/paper/000007.pdf>
4. C. Wang, G. Richard, and B. McFee, "Transfer learning and bias correction with pre-trained audio embeddings," in *Proc. 24th Int. Soc. Music Inf. Retrieval Conf. (ISMIR)*, Milan, Italy, 2023, pp. 64-70. [Online]. Available: <https://archives.ismir.net/ismir2023/paper/000006.pdf>
5. L. Ou, Y. Takahashi, and Y. Wang, "Lead instrument detection from multitrack music," in *Proc. IEEE Int. Conf. Acoust., Speech Signal Process. (ICASSP)*, 2025, pp. 1-5, doi: 10.1109/ICASSP49660.2025.10889928. [Online]. Available: <https://doi.org/10.1109/ICASSP49660.2025.10889928>

## AI Use Statement

ChatGPT was used to assist with grammar correction, improve sentence clarity, and refine the organisation of this project description. It also provided suggestions for the initial project and code structure. All technical decisions, references, code, and final content were reviewed and verified by the author.
