# Weakly Supervised Temporal Localisation of Musical Instruments

Ryan Hu · ELEC5305 · The University of Sydney

[Final report (PDF)](https://github.com/Decarre/elec5305-project-550275044/blob/main/output/pdf/ELEC5305_Project_Report_550275044_Slakh160.pdf) · [Final demonstration (MP4)](https://github.com/Decarre/elec5305-project-550275044/blob/main/output/video/ELEC5305_Slakh160_Presentation_Polished.mp4) · [Code and run instructions](https://github.com/Decarre/elec5305-project-550275044#run-the-demo)

The project tests whether a model trained to recognise instruments in five-second audio clips can also find when drums, bass, guitar and piano play. The final experiment uses 160 Slakh2100-redux tracks: 120 for training, 20 for validation and 20 for testing.

| Pooling | Original mixes | All three variants |
| --- | ---: | ---: |
| Mean | 68.9% | 68.2% |
| Max | 63.5% | 65.2% |
| Attention | 70.7% | 69.9% |

The table reports test frame-level macro balanced accuracy. Each model ran once. Max ranked highest on clip-level validation. The video shows a matched bass onset and a missed piano interval from the validation set.

[Saved test results](https://github.com/Decarre/elec5305-project-550275044/blob/main/results/slakh160/results.json) · [BabySlakh supplement (PDF)](https://github.com/Decarre/elec5305-project-550275044/blob/main/output/pdf/ELEC5305_Project_Supplement_550275044_V4.pdf)

Audio excerpts are from Slakh2100-redux by Ethan Manilow, Gordon Wichern, Prem Seetharaman and Jonathan Le Roux. [Dataset and attribution](https://zenodo.org/records/4599666)
