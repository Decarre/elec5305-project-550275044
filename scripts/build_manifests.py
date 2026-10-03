"""Build leakage-safe track and clip manifests from the metadata audit."""

import argparse
import csv
import json
from pathlib import Path

import soundfile as sf

from instrument_localization.manifest import (
    build_clip_manifest,
    build_track_manifest,
    manifest_summary,
    write_manifest,
)
from instrument_localization.splits import balanced_artist_split


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(args):
    audit_rows = read_rows(args.tracks_csv)
    artists = sorted({row["artist"] for row in audit_rows})
    if args.engineering_sample:
        split_by_artist = {artist: "engineering" for artist in artists}
    else:
        split_by_artist = balanced_artist_split(
            audit_rows, args.families, seed=args.seed, annotated_only_for_balance=True
        )
    tracks = build_track_manifest(
        audit_rows, split_by_artist, args.audio_root, args.reference_root, args.families,
        activation_root=args.activation_root,
    )
    if not args.allow_missing_audio and any(not row.audio_available for row in tracks):
        missing = sum(not row.audio_available for row in tracks)
        raise FileNotFoundError("{} manifest tracks have no mix WAV".format(missing))

    durations = {}
    for row in tracks:
        if row.audio_available:
            durations[row.track_id] = float(sf.info(row.audio_path).duration)
    clips = build_clip_manifest(tracks, durations, args.clip_duration, args.clip_hop)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(args.output_dir / "tracks.csv", tracks)
    if clips:
        write_manifest(args.output_dir / "clips.csv", clips)
    summary = manifest_summary(tracks, clips)
    summary.update({
        "engineering_sample": args.engineering_sample,
        "generalisation_valid": not args.engineering_sample and len(set(split_by_artist.values())) == 3,
        "target_families": args.families,
        "clip_duration": args.clip_duration,
        "clip_hop": args.clip_hop,
        "seed": args.seed,
    })
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks-csv", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument(
        "--activation-root", type=Path,
        help="Override the official v2 activation directory (useful for MedleyDB_sample).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families", nargs="+", default=["drums", "bass", "guitar", "piano", "strings"])
    parser.add_argument("--clip-duration", type=float, default=10.0)
    parser.add_argument("--clip-hop", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=5305)
    parser.add_argument("--allow-missing-audio", action="store_true")
    parser.add_argument(
        "--engineering-sample", action="store_true",
        help="Put all tracks in a non-generalisation engineering split.",
    )
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
