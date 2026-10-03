"""Audit official MedleyDB v1/v2 metadata without requiring the audio files.

The output is a *metadata* audit. Stem presence is not evidence of activity in
every short clip, and the split generated here is a plan until audio is checked.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

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


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def audit(reference_root, output_dir):
    package = reference_root / "medleydb"
    resources = package / "resources"
    metadata_dir = package / "data" / "Metadata"
    annotation_dir = package / "data" / "Annotations" / "Activation_Confidence" / "v2"
    track_ids = []
    version_by_track = {}
    for version in ("v1", "v2"):
        version_tracks = (resources / ("tracklist_" + version + ".txt")).read_text(encoding="utf-8").splitlines()
        track_ids.extend(version_tracks)
        version_by_track.update({track: version for track in version_tracks})
    if len(track_ids) != len(set(track_ids)):
        raise ValueError("Official v1/v2 track lists overlap")
    artist_index = json.loads((resources / "artist_index.json").read_text(encoding="utf-8"))
    rows = []
    for track_id in track_ids:
        metadata_path = metadata_dir / (track_id + "_METADATA.yaml")
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        instruments = sorted({str(stem.get("instrument", "")).lower() for stem in metadata["stems"].values()})
        families = sorted(name for name, members in FAMILIES.items() if members.intersection(instruments))
        annotation_path = annotation_dir / (track_id + "_ACTIVATION_CONF.lab")
        rows.append({
            "track_id": track_id,
            "artist": artist_index.get(track_id, str(metadata.get("artist", ""))),
            "version": version_by_track[track_id],
            "families": ";".join(families),
            "stem_instruments": ";".join(instruments),
            "activation_confidence_v2": annotation_path.is_file(),
        })

    # Group before any future clip segmentation. Stable hash makes the proposal
    # independent of filesystem enumeration order and does not use audio labels.
    artists = sorted({row["artist"] for row in rows})
    ordered = sorted(artists, key=lambda value: hashlib.sha256(("elec5305-2026:" + value).encode()).hexdigest())
    group_tracks = Counter(row["artist"] for row in rows)
    total = len(rows)
    cumulative = 0
    split_by_artist = {}
    for artist in ordered:
        midpoint = (cumulative + group_tracks[artist] / 2) / total
        split_by_artist[artist] = "train" if midpoint < 0.70 else "validation" if midpoint < 0.85 else "test"
        cumulative += group_tracks[artist]
    for row in rows:
        row["planned_split"] = split_by_artist[row["artist"]]

    class_rows = []
    for family in FAMILIES:
        matching = [row for row in rows if family in row["families"].split(";")]
        counts = Counter(row["planned_split"] for row in matching)
        class_rows.append({
            "family": family,
            "tracks": len(matching),
            "artists": len({row["artist"] for row in matching}),
            "annotated_tracks": sum(row["activation_confidence_v2"] for row in matching),
            "annotated_artists": len({row["artist"] for row in matching if row["activation_confidence_v2"]}),
            "train_tracks": counts["train"],
            "validation_tracks": counts["validation"],
            "test_tracks": counts["test"],
            "train_artists": len({row["artist"] for row in matching if row["planned_split"] == "train"}),
            "validation_artists": len({row["artist"] for row in matching if row["planned_split"] == "validation"}),
            "test_artists": len({row["artist"] for row in matching if row["planned_split"] == "test"}),
            "validation_annotated_tracks": sum(row["activation_confidence_v2"] for row in matching if row["planned_split"] == "validation"),
            "test_annotated_tracks": sum(row["activation_confidence_v2"] for row in matching if row["planned_split"] == "test"),
            "train_annotated_artists": len({row["artist"] for row in matching if row["planned_split"] == "train" and row["activation_confidence_v2"]}),
            "validation_annotated_artists": len({row["artist"] for row in matching if row["planned_split"] == "validation" and row["activation_confidence_v2"]}),
            "test_annotated_artists": len({row["artist"] for row in matching if row["planned_split"] == "test" and row["activation_confidence_v2"]}),
        })

    cooccurrence_rows = []
    for family_a in FAMILIES:
        for family_b in FAMILIES:
            cooccurrence_rows.append({
                "family_a": family_a,
                "family_b": family_b,
                "tracks": sum(
                    family_a in row["families"].split(";") and family_b in row["families"].split(";")
                    for row in rows
                ),
                "artists": len({
                    row["artist"] for row in rows
                    if family_a in row["families"].split(";") and family_b in row["families"].split(";")
                }),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "tracks.csv", rows, list(rows[0]))
    write_csv(output_dir / "class_coverage.csv", class_rows, list(class_rows[0]))
    write_csv(output_dir / "class_cooccurrence.csv", cooccurrence_rows, list(cooccurrence_rows[0]))
    summary = {
        "source": "marl/medleydb official v1/v2 track lists and metadata",
        "reference_commit": "537bb7bad0601fc58e4fff8404f71658b86ef60e",
        "tracks": total,
        "artists": len(artists),
        "activation_confidence_v2_files": sum(row["activation_confidence_v2"] for row in rows),
        "planned_split_tracks": dict(Counter(row["planned_split"] for row in rows)),
        "planned_split_artists": dict(Counter(split_by_artist.values())),
        "audio_audited": False,
        "note": "Stem metadata gives song-level candidate labels; no clip-level labels or audio checks are claimed.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True, help="Checkout of official marl/medleydb repository")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.reference_root, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
