from pathlib import Path

from instrument_localization.manifest import (
    build_clip_manifest,
    build_track_manifest,
    manifest_hash,
    manifest_summary,
)


def test_manifest_freezes_artist_split_before_clipping(tmp_path):
    reference = tmp_path / "reference"
    metadata = reference / "medleydb" / "data" / "Metadata"
    activation = reference / "medleydb" / "data" / "Annotations" / "Activation_Confidence" / "v2"
    audio = tmp_path / "audio"
    metadata.mkdir(parents=True)
    activation.mkdir(parents=True)
    rows = []
    split_by_artist = {}
    for index, split in enumerate(("train", "validation", "test")):
        track_id = "Artist{}_Song".format(index)
        artist = "Artist{}".format(index)
        rows.append({
            "track_id": track_id,
            "artist": artist,
            "version": "v1",
            "families": "drums;guitar",
        })
        split_by_artist[artist] = split
        (metadata / (track_id + "_METADATA.yaml")).write_text("stems: {}\n")
    tracks = build_track_manifest(
        rows, split_by_artist, audio, reference, ["drums", "guitar"]
    )
    clips = build_clip_manifest(
        tracks, {row.track_id: 25.0 for row in tracks}, 10.0, 5.0
    )
    assert len(clips) == 12
    assert {clip.split for clip in clips} == {"train", "validation", "test"}
    assert len(manifest_hash(tracks)) == 64
    summary = manifest_summary(tracks, clips)
    assert summary["artists"] == 3
    assert summary["clips"] == 12
    assert all(not row.audio_available for row in tracks)
