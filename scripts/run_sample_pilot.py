"""Run a deliberately limited weak-label overfit check on MedleyDB_sample.

The two public sample songs cannot support an artist-disjoint evaluation. This
script trains and evaluates on the same clips only to verify that feature,
weak-label, mean-pooling and AttentionMIC-style paths execute end to end. Its
metrics must be reported as training-set engineering diagnostics, never as
generalisation or research results.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from instrument_localization.models import AttentionInstrumentModel, BaselineInstrumentModel  # noqa: E402
from instrument_localization.annotations import read_activation_confidence  # noqa: E402
from instrument_localization.metrics import multilabel_metrics  # noqa: E402
from instrument_localization.taxonomy import family_for_instrument  # noqa: E402


CLASSES = ("guitar", "strings", "vocals")


def make_dataset(sample_root, clip_seconds, positive_fraction):
    features = []
    labels = []
    records = []
    references = []
    sample_rate = 22050
    samples_per_clip = int(sample_rate * clip_seconds)
    annotation_root = sample_root / "Annotations" / "Instrument_Activations" / "ACTIVATION_CONF"
    for track_dir in sorted((sample_root / "Audio").iterdir()):
        if not track_dir.is_dir():
            continue
        track_id = track_dir.name
        metadata = yaml.safe_load((track_dir / (track_id + "_METADATA.yaml")).read_text(encoding="utf-8"))
        waveform, _ = librosa.load(str(track_dir / metadata["mix_filename"]), sr=sample_rate, mono=True)
        times, stem_curves = read_activation_confidence(
            annotation_root / (track_id + "_ACTIVATION_CONF.lab")
        )
        family_curves = {name: [] for name in CLASSES}
        for stem_id, stem in metadata["stems"].items():
            family = family_for_instrument(stem["instrument"], CLASSES)
            if family is not None and stem_id in stem_curves:
                family_curves[family].append(stem_curves[stem_id])
        family_curves = {
            name: np.max(np.stack(curves), axis=0) if curves else np.zeros_like(times)
            for name, curves in family_curves.items()
        }
        clip_count = len(waveform) // samples_per_clip
        for clip_index in range(clip_count):
            start = clip_index * clip_seconds
            end = start + clip_seconds
            clip = waveform[clip_index * samples_per_clip:(clip_index + 1) * samples_per_clip]
            mel = librosa.feature.melspectrogram(
                y=clip, sr=sample_rate, n_fft=2048, hop_length=512, n_mels=64, power=2.0
            )
            log_mel = librosa.power_to_db(mel, ref=np.max)
            # Fixed physical scale: -80 dB to 0 dB becomes -1 to 0.
            features.append(np.clip(log_mel, -80.0, 0.0).astype(np.float32) / 80.0)
            in_clip = (times >= start) & (times < end)
            reference = np.stack([family_curves[name][in_clip] for name in CLASSES], axis=1)
            binary = (reference >= 0.5).mean(axis=0) >= positive_fraction
            labels.append(binary.astype(np.float32))
            records.append({"track_id": track_id, "clip_index": clip_index, "start_seconds": start, "end_seconds": end})
            references.append((times[in_clip] - start, reference))
    return torch.from_numpy(np.stack(features)), torch.from_numpy(np.stack(labels)), records, references


def train(model_name, inputs, targets, epochs, learning_rate, seed):
    torch.manual_seed(seed)
    model_type = BaselineInstrumentModel if model_name == "mean" else AttentionInstrumentModel
    model = model_type(n_mels=64, num_instruments=len(CLASSES), hidden_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    losses = []
    model.train()
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(len(inputs))
        epoch_loss = 0.0
        for start in range(0, len(inputs), 16):
            indices = permutation[start:start + 16]
            output = model(inputs[indices])
            loss = criterion(output["clip_logits"], targets[indices])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss) * len(indices)
        losses.append(epoch_loss / len(inputs))
    model.eval()
    with torch.no_grad():
        output = model(inputs)
        probabilities = output["clip_probabilities"].numpy()
    evaluated = multilabel_metrics(targets.numpy(), probabilities, 0.5, CLASSES)
    class_rows = []
    for index, row in enumerate(evaluated["per_class"]):
        class_rows.append({
            "class": row["class"],
            "positive_clips": int((targets.numpy()[:, index] >= 0.5).sum()),
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
        })
    micro_f1 = evaluated["micro"]["f1"]
    macro_f1 = evaluated["macro"]["f1"]
    return model, losses, class_rows, micro_f1, macro_f1


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(sample_root, output_dir, epochs, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    started = time.time()
    inputs, targets, records, references = make_dataset(sample_root, clip_seconds=2.0, positive_fraction=0.10)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = {}
    loss_rows = []
    metric_rows = []
    summaries = {}
    for model_name in ("mean", "attention"):
        model, losses, class_rows, micro_f1, macro_f1 = train(model_name, inputs, targets, epochs, 0.003, seed)
        models[model_name] = model
        loss_rows.extend({"model": model_name, "epoch": index + 1, "training_bce": value} for index, value in enumerate(losses))
        metric_rows.extend({"model": model_name, **row} for row in class_rows)
        summaries[model_name] = {
            "initial_training_bce": losses[0],
            "final_training_bce": losses[-1],
            "training_micro_f1": micro_f1,
            "training_macro_f1": macro_f1,
        }
    write_csv(output_dir / "loss_curves.csv", loss_rows)
    write_csv(output_dir / "training_metrics.csv", metric_rows)

    # Select a clip with the most within-clip vocal variation for signal QA.
    display_index = int(np.argmax([np.std(reference[:, 2]) for _, reference in references]))
    time_axis, reference = references[display_index]
    with torch.no_grad():
        mean_output = models["mean"](inputs[display_index:display_index + 1])
        attention_output = models["attention"](inputs[display_index:display_index + 1])
    frame_times = np.linspace(0.0, 2.0, mean_output["frame_logits"].shape[1], endpoint=False)
    figure, axes = plt.subplots(5, 1, figsize=(9, 8.5), sharex=True)
    axes[0].imshow(inputs[display_index].numpy(), aspect="auto", origin="lower", extent=(0, 2, 0, 64), cmap="magma")
    axes[0].set_ylabel("Mel bin")
    axes[0].set_title("Training-set sanity check: " + records[display_index]["track_id"] + " at " + str(records[display_index]["start_seconds"]) + " s")
    for class_index, name in enumerate(CLASSES):
        axes[1].plot(time_axis, reference[:, class_index], label=name)
        axes[2].plot(frame_times, torch.sigmoid(mean_output["frame_logits"])[0, :, class_index].numpy(), label=name)
        axes[3].plot(frame_times, attention_output["frame_probabilities"][0, :, class_index].numpy(), label=name)
        axes[4].plot(frame_times, attention_output["attention_weights"][0, :, class_index].numpy(), label=name)
    axes[1].set_ylabel("Reference\nconfidence")
    axes[2].set_ylabel("Mean model\nframe score")
    axes[3].set_ylabel("Attention model\nframe score")
    axes[4].set_ylabel("Attention\nweight")
    axes[4].set_xlabel("Time within 2 s clip (s)")
    axes[4].set_title("Per-class relative weights; each curve sums to 1 over the clip")
    for axis in axes[1:]:
        axis.legend(loc="upper right", ncol=3, fontsize=7)
    figure.tight_layout()
    figure.savefig(output_dir / "pilot_signals.png", dpi=160)
    plt.close(figure)

    summary = {
        "purpose": "training-set overfit/sanity check only; no held-out evaluation or generalisation claim",
        "source_tracks": sorted({record["track_id"] for record in records}),
        "clips": len(records),
        "clip_seconds": 2.0,
        "classes": list(CLASSES),
        "pooling_protocol": "controlled frame-probability pooling for both mean and attention models",
        "positive_label_rule": "activation confidence >=0.5 for at least 10% of annotation frames in the clip",
        "training_and_evaluation_set_are_identical": True,
        "epochs": epochs,
        "seed": seed,
        "models": summaries,
        "runtime_seconds": time.time() - started,
        "limitation": "Two songs and two artists cannot support artist-disjoint train/validation/test partitions; metrics only verify the implementation can fit this tiny sample.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seed", type=int, default=5305)
    args = parser.parse_args()
    print(json.dumps(run(args.sample_root, args.output_dir, args.epochs, args.seed), indent=2))


if __name__ == "__main__":
    main()
