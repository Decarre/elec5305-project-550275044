# ELEC5305 Project Proposal

## 1. Project Title

**Attention-Based Temporal Localisation of Musical Instruments in Polyphonic Music**

## 2. Student Information

- **Full Name:** Ryan Hu
- **Student ID (SID):** 550275044
- **GitHub Username:** Decarre
- **GitHub Repository:** <https://github.com/Decarre/elec5305-project-550275044>
- **GitHub Project Site:** <https://decarre.github.io/elec5305-project-550275044/>
- **Proposal word count:** 953 words (Sections 3-7, excluding headings and table labels)

## 3. Project Overview

This project will investigate how musical instruments can be identified and temporally localised in polyphonic music, where several instruments may be active at the same time. A conventional clip-level classifier can indicate that a song contains guitar or drums, but it does not necessarily show when each instrument enters or exits. This limits its usefulness for music analysis, arrangement visualisation, and content navigation. The proposed solution is a multi-label neural network that processes a log-mel spectrogram and assigns instrument-specific attention weights across time. These weights will be converted into predicted activity intervals and displayed as an instrument timeline. The central research question is: **Can an attention-based model trained with clip-level instrument labels estimate instrument entry and exit times accurately enough to outperform a conventional global-pooling baseline?**

## 4. Background and Motivation

Automatic recognition of instruments in ensemble recordings remains difficult because timbres overlap, quieter sources may be masked, and some instruments frequently occur together. MedleyDB provides multitrack mixtures, stems, metadata, and time-aligned instrument activations, making it suitable for evaluating temporal predictions [1]. OpenMIC-2018 provides a larger collection of ten-second polyphonic excerpts with partially observed multi-label annotations, but it does not provide precise onset and offset labels [2]. These datasets illustrate a central trade-off: strongly labelled data supports temporal evaluation but is comparatively small, while weakly labelled data is easier to collect at scale.

Gururani, Sharma, and Lerch showed that instrument-specific attention can improve weakly supervised multi-label recognition and can highlight relevant short-time regions [3]. However, an attention map should not automatically be treated as a correct explanation. Its temporal locations must be tested against independent frame-level annotations. Recent work also shows that instrument recognition can be affected by dataset identity and genre distribution, so apparently strong results may not generalise to new music [4]. More recent music representation models and benchmarks, including MARBLE and MERT, provide useful context for evaluating learned audio features [5, 6]. The topic was selected because it combines signal-processing concepts from ELEC5305, including framing, STFT analysis, spectral features, and temporal modelling, with a practical and visually interpretable music application.

## 5. Proposed Methodology

The primary tools will be MATLAB and Python. MATLAB course examples will be used to verify STFT framing and to construct a conventional feature baseline using quantities such as MFCCs, spectral centroid, spectral flux, spectral roll-off, and zero-crossing rate. Python, PyTorch, NumPy, and librosa will be used for the neural models, temporal post-processing, evaluation, and visualisation. Any course procedure adapted into the project will be identified in the code and report; the project-specific model design, experiments, and analysis will remain separate.

MedleyDB is the primary planned dataset because its stems and activation annotations allow reference entry and exit times to be generated. The final scope will be limited to approximately five sufficiently represented instrument families, tentatively drums, bass, guitar, piano or keyboard, and strings. Classes will be confirmed after a frequency audit. Songs, rather than randomly extracted windows, will be divided into training, validation, and test sets to prevent segments from the same recording leaking across splits. If practical, OpenMIC-2018 will provide a secondary clip-level experiment, but it is not required for the minimum viable project.

Audio will be resampled consistently and transformed into log-mel spectrograms using an STFT. The first neural baseline will use a convolutional frame encoder followed by global mean pooling and sigmoid multi-label outputs. The proposed model will retain the encoder but replace global pooling with instrument-specific temporal attention. It will output clip probabilities, frame probabilities, and a separate attention curve for each instrument. Thresholding, median or moving-average smoothing, and a minimum-duration rule will convert the temporal scores into activity intervals.

The baseline and attention model will be trained and evaluated under the same data splits. Clip-level performance will be measured using micro-F1, macro-F1, precision, recall, and per-instrument F1. Temporal performance will be evaluated using frame-level F1, event-level F1 under a stated tolerance, and mean onset and offset timing error. An ablation comparison will test global pooling, attention pooling, and attention with smoothing. Error analysis will include per-label binary confusion matrices, pairwise false-positive patterns, and instrument co-occurrence statistics. This will help distinguish genuine timbral confusion from errors caused by class imbalance, masking, or correlated arrangements. Recent frame-level attention research on lead-instrument detection provides a useful comparison while addressing a different single-lead formulation [7], and the recent HamNava study provides an additional multi-label dataset and evaluation reference [8].

## 6. Expected Outcomes

The minimum expected outcome is a working and documented prototype that accepts a compatible audio recording and produces instrument probabilities and a temporal activity plot. The project will deliver reproducible preprocessing, baseline and attention model definitions, saved experiment configurations, evaluation scripts, figures, and representative error examples. The main quantitative outcomes will be clip-level and frame-level F1 scores plus onset and offset errors. The main qualitative outcome will be an instrument timeline that can be compared with the reference activation annotations. The project is feasible within one semester because it is restricted to about five instrument families and one principal model comparison. Training a large foundation model, supporting arbitrary commercial songs, and performing full source separation are outside the required scope. The public GitHub repository will contain working code, setup instructions, the proposal, progress documentation, and final demonstration material where licensing permits.

## 7. Timeline (Weeks 1-13)

| Week | Task |
|---|---|
| 1-2 | Confirm the research question and restrict the target task. |
| 3-5 | Complete the literature review, obtain dataset access, audit labels, and define song-level splits. |
| 6-7 | Implement and test STFT/log-mel preprocessing and the conventional baseline. |
| 8-9 | Implement the attention model, training loop, and temporal interval extraction. |
| 10-11 | Tune thresholds, run ablations, evaluate timing accuracy, and analyse instrument confusions. |
| 12 | Prepare figures, GitHub documentation, reproducibility instructions, and the demonstration. |
| 13 | Complete the report, video, final verification, and submission. |

## 8. References

1. Bittner, R. M., Salamon, J., Tierney, M., Mauch, M., Cannam, C., & Bello, J. P. (2014). MedleyDB: A multitrack dataset for annotation-intensive MIR research. *Proceedings of ISMIR*, 155-160. <https://medleydb.weebly.com/>
2. Humphrey, E. J., Durand, S., & McFee, B. (2018). OpenMIC-2018: An open dataset for multiple instrument recognition. *Proceedings of ISMIR*, 438-444. <https://archives.ismir.net/ismir2018/paper/000248.pdf>
3. Gururani, S., Sharma, M., & Lerch, A. (2019). An attention mechanism for musical instrument recognition. *Proceedings of ISMIR*, 83-90. <https://archives.ismir.net/ismir2019/paper/000007.pdf>
4. Wang, C., Richard, G., & McFee, B. (2023). Transfer learning and bias correction with pre-trained audio embeddings. *Proceedings of ISMIR*, 64-71. <https://archives.ismir.net/ismir2023/paper/000006.pdf>
5. Yuan, R., et al. (2023). MARBLE: Music audio representation benchmark for universal evaluation. *Advances in Neural Information Processing Systems, 36*, 39626-39647. <https://proceedings.neurips.cc/paper_files/paper/2023/hash/7cbeec46f979618beafb4f46d8f39f36-Abstract-Datasets_and_Benchmarks.html>
6. Li, Y., et al. (2024). MERT: Acoustic music understanding model with large-scale self-supervised training. *International Conference on Learning Representations*. <https://proceedings.iclr.cc/paper_files/paper/2024/hash/33dffa2e3d2ab74a783d1a8c292f66d9-Abstract-Conference.html>
7. Ou, L., Takahashi, Y., & Wang, Y. (2025). Lead instrument detection from multitrack music. *Proceedings of IEEE ICASSP*, 1-5. <https://smcnus.comp.nus.edu.sg/archive/pdf/2025/2025_Lead_Instrument_Detection_from_Multitrack_Music_final.pdf>
8. Mohseni, P., BabaAli, B., & Asadi, H. (2025). HamNava: A dataset for multi-label instrument classification. *Transactions of the International Society for Music Information Retrieval, 8*(1), 236-247. <https://doi.org/10.5334/tismir.257>

## 9. Appendix: Planned System Flow

```text
Audio recording
      |
STFT and log-mel spectrogram
      |
Convolutional frame encoder
      |----------------------------|
Global-pooling baseline     Instrument-specific attention
      |                            |
Clip-level labels        Frame scores and attention curves
                                   |
                         Temporal smoothing and thresholding
                                   |
                      Instrument entry/exit timeline
```

No training results are claimed at the proposal stage. The existing repository framework has been checked with configuration, model-shape, attention-normalisation, and interval-extraction unit tests.
