"""Prepare and evaluate the full deduplicated Slakh2100 16 kHz release."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from instrument_localization.alignment import aggregate_time_grid
from instrument_localization.config import ExperimentConfig, load_config
from instrument_localization.events import event_metrics
from instrument_localization.features import checkpoint_features, PREPROCESSING_VERSION
from instrument_localization.inference import sha256
from instrument_localization.metrics import multilabel_metrics, select_balanced_thresholds
from instrument_localization.slakh import SLAKH_CLASSES, assert_group_disjoint, discover_tracks, file_sha256, normalise_inst_class
from instrument_localization.training import load_checkpoint, train_arrays

BASE = load_config(ROOT / "configs/babyslakh_attention.yaml")
PARTITIONS = ("train", "validation", "test")
VARIANTS = ("original", "absence", "transition")
SAMPLES_PER_BIN = 8000
CLIP_SAMPLES = 80000


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_audio(path):
    import soundfile as sf

    samples, rate = sf.read(str(path), dtype="float32")
    if samples.ndim != 1:
        raise ValueError("Expected mono audio: " + str(path))
    if rate != 16000:
        from math import gcd
        from scipy.signal import resample_poly

        divisor = gcd(rate, 16000)
        samples = resample_poly(samples, 16000 // divisor, rate // divisor)
    return samples


def family_sources(track, stems, length):
    sources = np.zeros((length, len(SLAKH_CLASSES) + 1), dtype=np.float32)
    for stem_id, info in stems.items():
        if info.get("audio_rendered") is False:
            continue
        stem = track / "stems" / (stem_id + ".flac")
        if not stem.is_file():
            stem = stem.with_suffix(".wav")
        samples = read_audio(stem)
        family = normalise_inst_class(info.get("inst_class", ""))
        channel = SLAKH_CLASSES.index(family) if family else len(SLAKH_CLASSES)
        count = min(length, len(samples))
        sources[:count, channel] += samples[:count]
    return sources


def controlled_audio(sources, gates):
    envelope = np.repeat(gates, SAMPLES_PER_BIN, axis=0)
    for boundary, family in zip(*np.nonzero(np.diff(gates, axis=0))):
        sample = (boundary + 1) * SAMPLES_PER_BIN
        envelope[sample - 40:sample + 40, family] = np.linspace(
            gates[boundary, family], gates[boundary + 1, family], 80)
    return (sources[:, :4] * envelope).sum(axis=1) + sources[:, 4]


def prepare(dataset, output, smoke, subset_manifest=None):
    import soundfile as sf
    from prepare_babyslakh import mix_path, note_intervals, reference_grid
    from prepare_controlled_babyslakh import control_gates

    discovered = discover_tracks(dataset)
    tracks = {partition: [track for track in discovered if track.parent.name == partition]
              for partition in PARTITIONS}
    if min(map(len, tracks.values())) < 1:
        raise ValueError("Expected the official train, validation and test directories.")
    if subset_manifest:
        manifest = json.loads(subset_manifest.read_text(encoding="utf-8"))
        expected = manifest["tracks"]
        if {name: sorted(item.name for item in items) for name, items in tracks.items()} != expected:
            raise ValueError("Downloaded tracks do not match the fixed subset manifest.")
    elif not smoke and sum(map(len, tracks.values())) != 1710:
        raise ValueError("Expected all 1,710 non-omitted tracks in the deduplicated release.")
    if smoke:
        tracks = {partition: items[:1] for partition, items in tracks.items()}
    groups = {track.name: file_sha256(track / "all_src.mid") for items in tracks.values() for track in items}
    split = {track.name: partition for partition, items in tracks.items() for track in items}
    if len(groups) != sum(map(len, tracks.values())):
        raise ValueError("Duplicate track IDs across partitions.")
    assert_group_disjoint(groups, split)
    output.mkdir(parents=True, exist_ok=True)
    identity = {"dataset": "Slakh2100-redux subset" if subset_manifest else "Slakh2100-redux 16 kHz",
                "subset_manifest_sha256": sha256(subset_manifest) if subset_manifest else None, "smoke": smoke,
                "preprocessing": PREPROCESSING_VERSION, "groups": groups, "splits": split,
                "preparer_sha256": sha256(__file__)}
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Prepared data belongs to another run. Use a new output directory.")
    write_json(identity_path, identity)
    counts = {}
    for partition, items in tracks.items():
        path = output / partition
        path.mkdir(exist_ok=True)
        if (path / "complete.json").exists():
            counts[partition] = json.loads((path / "complete.json").read_text())
            continue
        infos = [sf.info(str(mix_path(track))) for track in items]
        clips = [((info.frames * 16000 + info.samplerate - 1) // info.samplerate)
                 // CLIP_SAMPLES for info in infos]
        if smoke:
            clips = [min(8, value) for value in clips]
        count = sum(clips) * len(VARIANTS)
        progress = path / "progress.json"
        finished = json.loads(progress.read_text()) if progress.exists() else 0
        mode = "r+" if finished else "w+"
        features = np.lib.format.open_memmap(path / "features.npy", mode, np.float32, (count, BASE.n_mels, 251))
        labels = np.lib.format.open_memmap(path / "labels.npy", mode, np.float32, (count, 4))
        if partition != "train":
            references = np.lib.format.open_memmap(path / "references.npy", mode, np.uint8, (count, 10, 4))
        index = sum(clips[:finished]) * len(VARIANTS)
        for track_index in range(finished, len(items)):
            track, num_clips = items[track_index], clips[track_index]
            metadata = yaml.safe_load((track / "metadata.yaml").read_text(encoding="utf-8"))
            audio = read_audio(mix_path(track))
            sources = family_sources(track, metadata["stems"], len(audio))
            intervals = note_intervals(track, metadata)
            for clip in range(num_clips):
                start = clip * CLIP_SAMPLES
                stop = start + CLIP_SAMPLES
                truth = reference_grid(intervals, clip * 5.0, 5, 0.5)
                for variant in VARIANTS:
                    seed = int(hashlib.sha256((track.name + str(clip) + variant + "protocol-v2").encode()).hexdigest()[:8], 16)
                    gates = control_gates(variant, np.random.RandomState(seed))
                    waveform = audio[start:stop] if variant == "original" else controlled_audio(sources[start:stop], gates)
                    features[index] = checkpoint_features(waveform, BASE)
                    labels[index] = (truth * gates).any(axis=0)
                    if partition != "train":
                        references[index] = truth * gates
                    index += 1
            print(partition, track.name, index, "/", count, flush=True)
            features.flush()
            labels.flush()
            if partition != "train":
                references.flush()
            write_json(progress, track_index + 1)
        counts[partition] = {"tracks": len(items), "clips_with_variants": count}
        write_json(path / "complete.json", counts[partition])
    write_json(output / "summary.json", counts)


def scores(model, features, config, device):
    clip, frame = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), config.batch_size):
            batch = torch.from_numpy(features[start:start + config.batch_size]).to(device)
            result = model(batch)
            clip.append(result["clip_probabilities"].cpu().numpy())
            frame.append(aggregate_time_grid(result["frame_probabilities"].cpu().numpy()))
    return np.concatenate(clip), np.concatenate(frame)


def train(prepared, output, device, seeds, epochs):
    identity = json.loads((prepared / "identity.json").read_text())
    summary = json.loads((prepared / "summary.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"data": identity, "counts": summary, "epochs": epochs, "seeds": seeds,
                "device": device, "selection": "validation clip macro balanced accuracy",
                "evaluation": "frame thresholds from validation; original and controlled variants reported",
                "training": "clip labels only", "runner_sha256": sha256(__file__),
                "trainer_sha256": sha256(ROOT / "src/instrument_localization/training.py")}
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Results directory belongs to another experiment.")
    write_json(protocol_path, protocol)
    arrays = {part: [np.load(prepared / part / (name + ".npy"), mmap_mode="c")
                     for name in ("features", "labels")] for part in PARTITIONS}
    vr, tr = [np.load(prepared / part / "references.npy", mmap_mode="r") for part in PARTITIONS[1:]]
    test_labels = arrays["test"][1]
    always_active = multilabel_metrics(tr, np.ones(tr.shape, dtype=np.uint8))
    write_json(output / "always_active_baseline.json", always_active)
    results = []
    for pooling in ("baseline", "max", "attention"):
        for seed in seeds:
            run = output / (pooling + "_" + str(seed))
            final = run / "result.json"
            if final.exists():
                results.append(json.loads(final.read_text()))
                continue
            config = ExperimentConfig(**{**BASE.to_dict(), "model_type": pooling, "seed": seed,
                                         "epochs": epochs}).validate()
            print("Training", pooling, seed, flush=True)
            if not (run / "validation_summary.json").exists():
                train_arrays(*arrays["train"], *arrays["validation"], config, run, device=device,
                             selection_objective="balanced_accuracy")
            checkpoint = run / "best_checkpoint.pt"
            model, _, state = load_checkpoint(checkpoint, device)
            _, validation_frame = scores(model, arrays["validation"][0], config, device)
            thresholds = select_balanced_thresholds(vr, validation_frame)
            test_clip, test_frame = scores(model, arrays["test"][0], config, device)
            result = {"pooling": pooling, "seed": seed, "validation": state["validation_metrics"],
                      "clip": multilabel_metrics(test_labels, test_clip, state["validation_thresholds"]),
                      "frame": multilabel_metrics(tr, test_frame, thresholds),
                      "events": event_metrics(tr, test_frame, thresholds), "variants": {}}
            for offset, variant in enumerate(VARIANTS):
                section = slice(offset, None, len(VARIANTS))
                result["variants"][variant] = {
                    "frame": multilabel_metrics(tr[section], test_frame[section], thresholds),
                    "events": event_metrics(tr[section], test_frame[section], thresholds)}
            write_json(run / "calibration.json", {"checkpoint_sha256": sha256(checkpoint),
                       "frame_thresholds": thresholds.tolist(), "preprocessing_version": PREPROCESSING_VERSION})
            write_json(final, result)
            results.append(result)
    write_json(output / "results.json", results)
    comparison = {}
    for pooling in ("baseline", "max", "attention"):
        runs = [result for result in results if result["pooling"] == pooling]
        overall = [result["frame"]["macro"]["balanced_accuracy"] for result in runs]
        original = [result["variants"]["original"]["frame"]["macro"]["balanced_accuracy"]
                    for result in runs]
        comparison[pooling] = {"controlled_frame_ba": float(np.mean(overall)),
                               "controlled_frame_ba_seed_sd": float(np.std(overall)),
                               "original_frame_ba": float(np.mean(original))}
    write_json(output / "comparison.json", comparison)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("dataset", type=Path)
    prep.add_argument("output", type=Path)
    prep.add_argument("--smoke", action="store_true")
    prep.add_argument("--subset-manifest", type=Path)
    fit = commands.add_parser("train")
    fit.add_argument("prepared", type=Path)
    fit.add_argument("output", type=Path)
    fit.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    fit.add_argument("--seeds", type=int, nargs="+", default=[5305, 5306, 5307])
    fit.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.command == "prepare":
        prepare(args.dataset, args.output, args.smoke, args.subset_manifest)
    else:
        train(args.prepared, args.output, args.device, args.seeds, args.epochs)


if __name__ == "__main__":
    main()
