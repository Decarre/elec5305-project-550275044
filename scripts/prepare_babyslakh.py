"""Prepare leakage-safe BabySlakh features and separate temporal references."""

import argparse
import csv
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from instrument_localization.slakh import (  # noqa: E402
    SLAKH_CLASSES,
    assert_group_disjoint,
    discover_tracks,
    file_sha256,
    grouped_track_split,
    midi_path,
    normalise_inst_class,
)
from instrument_localization.features import checkpoint_features
from instrument_localization.config import ExperimentConfig


def mix_path(track_dir):
    for name in ("mix.wav", "mix.flac"):
        candidate = track_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError("missing mix audio in " + str(track_dir))


def note_intervals(track_dir, metadata):
    intervals = {name: [] for name in SLAKH_CLASSES}
    for stem_id, stem in (metadata.get("stems") or {}).items():
        family = normalise_inst_class(stem.get("inst_class", ""))
        path = midi_path(track_dir, stem_id)
        if not family or not path.exists():
            continue
        midi = pretty_midi.PrettyMIDI(str(path))
        for instrument in midi.instruments:
            intervals[family].extend((note.start, note.end) for note in instrument.notes)
    return intervals


def reference_grid(intervals, start, clip_seconds, grid_seconds):
    bins = int(round(clip_seconds / grid_seconds))
    reference = np.zeros((bins, len(SLAKH_CLASSES)), dtype=np.float32)
    for bin_index in range(bins):
        left = start + bin_index * grid_seconds
        right = left + grid_seconds
        for class_index, name in enumerate(SLAKH_CLASSES):
            reference[bin_index, class_index] = any(
                note_start < right and note_end > left
                for note_start, note_end in intervals[name]
            )
    return reference


def write_rows(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(dataset_root, output_dir, clip_seconds=5.0, grid_seconds=0.5, seed=5305):
    tracks = discover_tracks(dataset_root)
    if not tracks:
        raise ValueError("no BabySlakh track directories found")
    track_to_group = {}
    for track in tracks:
        midi = track / "all_src.mid"
        if not midi.exists():
            raise FileNotFoundError("missing all_src.mid in " + str(track))
        track_to_group[track.name] = file_sha256(midi)
    split = grouped_track_split(track_to_group, seed)
    assert_group_disjoint(track_to_group, split)
    buffers = {
        name: {"features": [], "labels": [], "references": [], "track_ids": [], "starts": []}
        for name in ("train", "validation", "test")
    }
    manifest = []
    for track in tracks:
        metadata = yaml.safe_load((track / "metadata.yaml").read_text(encoding="utf-8"))
        waveform, _ = librosa.load(str(mix_path(track)), sr=16000, mono=True)
        intervals = note_intervals(track, metadata)
        partition = split[track.name]
        clip_samples = int(16000 * clip_seconds)
        clip_count = len(waveform) // clip_samples
        for clip_index in range(clip_count):
            start = clip_index * clip_seconds
            clip = waveform[clip_index * clip_samples : (clip_index + 1) * clip_samples]
            feature = checkpoint_features(clip, ExperimentConfig(sample_rate=16000, n_fft=1024, hop_length=320))
            reference = reference_grid(intervals, start, clip_seconds, grid_seconds)
            label = reference.any(axis=0).astype(np.float32)
            target = buffers[partition]
            target["features"].append(feature)
            target["labels"].append(label)
            target["references"].append(reference)
            target["track_ids"].append(track.name)
            target["starts"].append(start)
            manifest.append({
                "track_id": track.name,
                "duplicate_group_sha256": track_to_group[track.name],
                "split": partition,
                "clip_index": clip_index,
                "start_seconds": start,
                "duration_seconds": clip_seconds,
                **{"label_" + name: int(label[index]) for index, name in enumerate(SLAKH_CLASSES)},
            })
    output_dir.mkdir(parents=True, exist_ok=True)
    for partition, values in buffers.items():
        if not values["features"]:
            raise ValueError("empty partition: " + partition)
        np.savez_compressed(
            str(output_dir / (partition + ".npz")),
            features=np.stack(values["features"]),
            labels=np.stack(values["labels"]),
        )
        np.savez_compressed(
            str(output_dir / (partition + "_temporal.npz")),
            references=np.stack(values["references"]),
            track_ids=np.asarray(values["track_ids"]),
            start_seconds=np.asarray(values["starts"], dtype=np.float32),
        )
    write_rows(output_dir / "clip_manifest.csv", manifest)
    split_tracks = {
        name: sorted(track_id for track_id, value in split.items() if value == name)
        for name in ("train", "validation", "test")
    }
    summary = {
        "dataset": "BabySlakh 16 kHz",
        "classes": list(SLAKH_CLASSES),
        "track_count": len(tracks),
        "duplicate_group_count": len(set(track_to_group.values())),
        "split_tracks": split_tracks,
        "split_clip_counts": {name: len(values["features"]) for name, values in buffers.items()},
        "clip_seconds": clip_seconds,
        "temporal_grid_seconds": grid_seconds,
        "seed": seed,
        "leakage_check": "duplicate MIDI SHA-256 groups are disjoint",
        "training_npz_contains_temporal_reference": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-seconds", type=float, default=5.0)
    parser.add_argument("--grid-seconds", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=5305)
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_root, args.output_dir, args.clip_seconds, args.grid_seconds, args.seed), indent=2))


if __name__ == "__main__":
    main()
