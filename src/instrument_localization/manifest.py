"""Track/clip manifests with explicit artist-level leakage barriers."""

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from .splits import assert_artist_disjoint


@dataclass(frozen=True)
class TrackManifestRow:
    track_id: str
    artist: str
    version: str
    split: str
    families: Sequence[str]
    audio_path: str
    metadata_path: str
    activation_path: str
    audio_available: bool
    activation_available: bool

    def to_dict(self) -> Dict[str, object]:
        row = asdict(self)
        row["families"] = ";".join(self.families)
        return row


@dataclass(frozen=True)
class ClipManifestRow:
    clip_id: str
    track_id: str
    artist: str
    split: str
    start_seconds: float
    end_seconds: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _find_audio(audio_root: Path, track_id: str) -> Path:
    candidates = (
        audio_root / track_id / (track_id + "_MIX.wav"),
        audio_root / "Audio" / track_id / (track_id + "_MIX.wav"),
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def build_track_manifest(
    audit_rows: Iterable[Mapping[str, object]],
    split_by_artist: Mapping[str, str],
    audio_root: Path,
    reference_root: Path,
    target_families: Sequence[str],
    activation_root: Optional[Path] = None,
) -> List[TrackManifestRow]:
    """Join audited metadata with local audio/annotation availability."""

    audio_root = Path(audio_root)
    package = Path(reference_root) / "medleydb"
    metadata_root = package / "data" / "Metadata"
    activation_root = Path(activation_root) if activation_root else (
        package / "data" / "Annotations" / "Activation_Confidence" / "v2"
    )
    rows = []
    for source in audit_rows:
        track_id = str(source["track_id"])
        artist = str(source["artist"])
        if artist not in split_by_artist:
            raise ValueError("missing split for artist: " + artist)
        source_families = source.get("families", "")
        if isinstance(source_families, str):
            source_families = source_families.split(";") if source_families else []
        families = tuple(family for family in target_families if family in source_families)
        audio_path = _find_audio(audio_root, track_id)
        metadata_path = metadata_root / (track_id + "_METADATA.yaml")
        activation_path = activation_root / (track_id + "_ACTIVATION_CONF.lab")
        rows.append(TrackManifestRow(
            track_id=track_id,
            artist=artist,
            version=str(source.get("version", "")),
            split=str(split_by_artist[artist]),
            families=families,
            audio_path=str(audio_path),
            metadata_path=str(metadata_path),
            activation_path=str(activation_path),
            audio_available=audio_path.is_file(),
            activation_available=activation_path.is_file(),
        ))
    validate_track_manifest(rows)
    return sorted(rows, key=lambda row: row.track_id)


def validate_track_manifest(rows: Sequence[TrackManifestRow]) -> None:
    if not rows:
        raise ValueError("track manifest is empty")
    ids = [row.track_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("track manifest contains duplicate track IDs")
    assert_artist_disjoint(
        {"artist": row.artist, "split": row.split} for row in rows
    )


def build_clip_manifest(
    tracks: Sequence[TrackManifestRow],
    durations: Mapping[str, float],
    clip_duration: float,
    hop_duration: float,
) -> List[ClipManifestRow]:
    """Create windows after track splits are frozen; no temporal labels are exposed."""

    validate_track_manifest(tracks)
    if clip_duration <= 0 or hop_duration <= 0:
        raise ValueError("clip duration and hop must be positive")
    clips = []
    for track in tracks:
        if track.track_id not in durations:
            continue
        duration = float(durations[track.track_id])
        if duration < clip_duration:
            continue
        index = 0
        start = 0.0
        while start + clip_duration <= duration + 1e-9:
            clips.append(ClipManifestRow(
                clip_id="{}__{:06d}".format(track.track_id, index),
                track_id=track.track_id,
                artist=track.artist,
                split=track.split,
                start_seconds=round(start, 6),
                end_seconds=round(start + clip_duration, 6),
            ))
            index += 1
            start = index * hop_duration
    return clips


def write_manifest(path: Path, rows: Sequence[object]) -> None:
    if not rows:
        raise ValueError("cannot write an empty manifest")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dictionaries = [row.to_dict() for row in rows]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dictionaries[0]))
        writer.writeheader()
        writer.writerows(dictionaries)


def manifest_hash(rows: Sequence[object]) -> str:
    """Return a stable hash for recording the exact split/window definition."""

    canonical = "\n".join(
        json.dumps(row.to_dict(), sort_keys=True, separators=(",", ":")) for row in rows
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def manifest_summary(
    tracks: Sequence[TrackManifestRow], clips: Optional[Sequence[ClipManifestRow]] = None
) -> Dict[str, object]:
    split_tracks = {}
    split_artists = {}
    for split in sorted({row.split for row in tracks}):
        selected = [row for row in tracks if row.split == split]
        split_tracks[split] = len(selected)
        split_artists[split] = len({row.artist for row in selected})
    return {
        "tracks": len(tracks),
        "artists": len({row.artist for row in tracks}),
        "audio_available": sum(row.audio_available for row in tracks),
        "activation_available": sum(row.activation_available for row in tracks),
        "tracks_by_split": split_tracks,
        "artists_by_split": split_artists,
        "track_manifest_sha256": manifest_hash(tracks),
        "clips": len(clips) if clips is not None else None,
        "clip_manifest_sha256": manifest_hash(clips) if clips else None,
    }
