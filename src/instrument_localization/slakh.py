"""BabySlakh/Slakh helpers with duplicate-group leakage protection."""

import hashlib
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


SLAKH_CLASSES = ("drums", "bass", "guitar", "piano")


def normalise_inst_class(value: str) -> str:
    """Map Slakh metadata instrument classes to the four controlled families."""

    text = str(value or "").strip().lower()
    aliases = {
        "drum": "drums",
        "drums": "drums",
        "percussion": "drums",
        "bass": "bass",
        "guitar": "guitar",
        "piano": "piano",
        "keyboard": "piano",
    }
    return aliases.get(text, "")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def grouped_track_split(
    track_to_group: Mapping[str, str], seed: int = 5305
) -> Dict[str, str]:
    """Return a deterministic 60/20/20 split without crossing group hashes."""

    groups: Dict[str, List[str]] = {}
    for track_id, group in track_to_group.items():
        groups.setdefault(group, []).append(track_id)
    ordered = sorted(groups)
    random.Random(seed).shuffle(ordered)
    count = len(ordered)
    if count < 3:
        raise ValueError("at least three duplicate groups are required")
    validation_count = max(1, round(count * 0.2))
    test_count = max(1, round(count * 0.2))
    train_count = count - validation_count - test_count
    if train_count < 1:
        raise ValueError("split would contain no training groups")
    group_split = {}
    for group in ordered[:train_count]:
        group_split[group] = "train"
    for group in ordered[train_count : train_count + validation_count]:
        group_split[group] = "validation"
    for group in ordered[train_count + validation_count :]:
        group_split[group] = "test"
    return {track_id: group_split[group] for track_id, group in track_to_group.items()}


def assert_group_disjoint(track_to_group: Mapping[str, str], split: Mapping[str, str]) -> None:
    seen: Dict[str, str] = {}
    for track_id, group in track_to_group.items():
        partition = split[track_id]
        if group in seen and seen[group] != partition:
            raise ValueError("duplicate MIDI group crosses dataset partitions")
        seen[group] = partition


def discover_tracks(root: Path) -> List[Path]:
    """Find Slakh track directories regardless of archive wrapper layout."""

    tracks = []
    for metadata in Path(root).rglob("metadata.yaml"):
        if metadata.parent.name.lower().startswith("track"):
            tracks.append(metadata.parent)
    return sorted(set(tracks), key=lambda path: path.name)


def midi_path(track_dir: Path, stem_id: str) -> Path:
    candidates = [
        track_dir / "MIDI" / (stem_id + ".mid"),
        track_dir / "MIDI" / (stem_id + ".midi"),
        track_dir / "midi" / (stem_id + ".mid"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]
