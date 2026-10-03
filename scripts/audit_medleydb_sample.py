"""Validate the official two-track MedleyDB sample and make a QA figure.

This is an engineering/data-alignment check, not a model-performance study.
The source audio remains outside the repository; only CSV/JSON summaries and a
derived figure are written to the requested output directory.
"""

import argparse
import csv
import json
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import yaml


FAMILIES = {
    "drums": {"drum set", "drum machine", "kick drum", "snare drum", "bass drum", "toms"},
    "bass": {"electric bass", "double bass"},
    "guitar": {
        "acoustic guitar", "clean electric guitar", "distorted electric guitar",
        "slide guitar", "lap steel guitar",
    },
    "piano": {"piano", "tack piano", "electric piano"},
    "strings": {
        "violin", "viola", "cello", "violin section", "viola section",
        "cello section", "string section",
    },
}


def family_for(instrument):
    instrument = instrument.lower()
    return next((name for name, members in FAMILIES.items() if instrument in members), "other")


def read_activation(path):
    values = np.genfromtxt(str(path), delimiter=",", names=True, dtype=np.float64)
    columns = list(values.dtype.names)
    return values["time"], {column: values[column] for column in columns if column != "time"}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit(sample_root, output_dir, threshold=0.5):
    audio_root = sample_root / "Audio"
    annotation_root = sample_root / "Annotations" / "Instrument_Activations" / "ACTIVATION_CONF"
    track_rows = []
    stem_rows = []
    plot_data = None

    for track_dir in sorted(path for path in audio_root.iterdir() if path.is_dir()):
        track_id = track_dir.name
        metadata = yaml.safe_load((track_dir / (track_id + "_METADATA.yaml")).read_text(encoding="utf-8"))
        mix_path = track_dir / metadata["mix_filename"]
        mix_info = sf.info(str(mix_path))
        times, activations = read_activation(annotation_root / (track_id + "_ACTIVATION_CONF.lab"))
        annotated_duration = float(times[-1]) if len(times) else 0.0
        annotation_hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
        annotated_stems = set(activations)
        expected_stems = {key for key in metadata["stems"] if metadata["stems"][key]["instrument"].lower() != "main system"}
        track_rows.append({
            "track_id": track_id,
            "artist": metadata["artist"],
            "mix_duration_seconds": round(float(mix_info.duration), 4),
            "sample_rate_hz": mix_info.samplerate,
            "channels": mix_info.channels,
            "metadata_stems": len(metadata["stems"]),
            "expected_annotated_stems": len(expected_stems),
            "activation_columns": len(annotated_stems),
            "activation_rows": len(times),
            "activation_hop_seconds": round(annotation_hop, 6),
            "annotation_duration_seconds": round(annotated_duration, 4),
            "duration_difference_seconds": round(float(mix_info.duration) - annotated_duration, 4),
            "columns_match_expected": annotated_stems == expected_stems,
        })

        family_curves = {}
        for stem_id, stem in metadata["stems"].items():
            if stem_id not in activations:
                continue
            stem_path = track_dir / metadata["stem_dir"] / stem["filename"]
            stem_info = sf.info(str(stem_path))
            family = family_for(stem["instrument"])
            stem_rows.append({
                "track_id": track_id,
                "stem_id": stem_id,
                "instrument": stem["instrument"],
                "family": family,
                "duration_seconds": round(float(stem_info.duration), 4),
                "sample_rate_hz": stem_info.samplerate,
                "active_fraction_at_0_5": round(float(np.mean(activations[stem_id] >= threshold)), 6),
                "mean_activation_confidence": round(float(np.mean(activations[stem_id])), 6),
            })
            family_curves.setdefault(family, []).append(activations[stem_id])
        family_curves = {family: np.max(np.stack(curves), axis=0) for family, curves in family_curves.items()}
        if track_id == "Phoenix_ScotchMorris":
            plot_data = (track_id, mix_path, times, family_curves)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "tracks.csv", track_rows)
    write_csv(output_dir / "stems.csv", stem_rows)
    summary = {
        "source": "official MedleyDB sample from Zenodo record 1438309",
        "tracks": len(track_rows),
        "artists": len({row["artist"] for row in track_rows}),
        "stems_with_activation_columns": len(stem_rows),
        "activation_threshold_for_summary": threshold,
        "all_annotation_columns_match_expected_non_main_system_stems": all(row["columns_match_expected"] for row in track_rows),
        "purpose": "audio and annotation alignment QA; no trained-model or generalisation claim",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    if plot_data is not None:
        track_id, mix_path, times, family_curves = plot_data
        duration = min(60.0, float(times[-1]))
        waveform, sample_rate = librosa.load(str(mix_path), sr=22050, mono=True, duration=duration)
        log_mel = librosa.power_to_db(
            librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=64),
            ref=np.max,
        )
        fig, axes = plt.subplots(1 + len(family_curves), 1, figsize=(10, 2.2 + 1.2 * len(family_curves)), sharex=True)
        axes = np.atleast_1d(axes)
        image = librosa.display.specshow(log_mel, sr=sample_rate, hop_length=512, x_axis="time", y_axis="mel", ax=axes[0])
        axes[0].set(title=track_id + ": first 60 s log-mel spectrogram")
        fig.colorbar(image, ax=axes[0], format="%+2.0f dB")
        for axis, family in zip(axes[1:], sorted(family_curves)):
            keep = times <= duration
            axis.plot(times[keep], family_curves[family][keep], linewidth=1.0)
            axis.axhline(threshold, color="tab:red", linestyle="--", linewidth=0.8, label="0.5 summary threshold")
            axis.set_ylim(-0.02, 1.02)
            axis.set_ylabel(family)
            axis.legend(loc="upper right", fontsize=7)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle("Official stem-derived activation confidence (not manual frame labels)")
        fig.tight_layout()
        fig.savefig(output_dir / "sample_alignment.png", dpi=160)
        plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(audit(args.sample_root, args.output_dir, args.threshold), indent=2))


if __name__ == "__main__":
    main()
