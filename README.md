# Weakly Supervised Temporal Localisation of Musical Instruments

Ryan Hu · ELEC5305 · The University of Sydney

[Project website](https://decarre.github.io/elec5305-project-550275044/) · [Final report](output/pdf/ELEC5305_Project_Report_550275044_Slakh160.pdf) · [Final video](output/video/ELEC5305_Slakh160_Presentation_Polished.mp4)

The project compares mean, max and attention pooling for instrument recognition and timing in polyphonic music. The 160-track Slakh2100-redux experiment uses 120 training, 20 validation and 20 test tracks. Attention reached 70.7% frame-level macro balanced accuracy on original test mixes and 69.9% across all three variants. The earlier BabySlakh experiment is documented in the [supplement](output/pdf/ELEC5305_Project_Supplement_550275044_V4.pdf).

## Run the demo

Clone the repository and run these commands from its root in PowerShell.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[audio,ml]"
.venv/Scripts/python.exe demo/serve.py --open
```

The browser opens at http://127.0.0.1:5306/. Choose an included validation clip and click **Run prediction**. The app also accepts local WAV files. It uses the trained attention checkpoint and its validation thresholds.

## Reproduce the 160-track experiment

Use a Python environment with the audio and ML dependencies. The first command downloads about 8.3 GB. The runner supports CUDA and CPU.

```powershell
New-Item -ItemType Directory -Force path/to/slakh160
Copy-Item results/slakh160/manifest.json path/to/slakh160/manifest.json
python scripts/download_slakh_subset.py path/to/slakh160
python scripts/run_slakh2100.py prepare path/to/slakh160/audio path/to/slakh160/prepared --subset-manifest path/to/slakh160/manifest.json
python scripts/run_slakh2100.py train path/to/slakh160/prepared path/to/slakh160/results --device cuda --seeds 5305
```

[Track and file manifest](results/slakh160/manifest.json) · [Evaluation protocol](results/slakh160/protocol.json) · [Full results](results/slakh160/results.json) · [Audio attribution](output/demo_slakh160/ATTRIBUTION.md)

The report source is [report/native/main.tex](report/native/main.tex). Compile it with Tectonic, or run pdfLaTeX twice from the report/native directory.
