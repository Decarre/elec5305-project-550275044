"""Deterministic, artist-disjoint and class-aware dataset splitting."""

import hashlib
from collections import Counter, defaultdict
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence


DEFAULT_FRACTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}


def _families(row: Mapping[str, object]) -> Sequence[str]:
    value = row.get("families", "")
    if isinstance(value, str):
        return tuple(item for item in value.split(";") if item)
    return tuple(str(item) for item in value)


def balanced_artist_split(
    rows: Iterable[Mapping[str, object]],
    target_families: Sequence[str],
    fractions: Mapping[str, float] = DEFAULT_FRACTIONS,
    seed: int = 5305,
    annotated_only_for_balance: bool = True,
) -> Dict[str, str]:
    """Assign each artist to one split while approximately balancing classes.

    Track and class targets are optimized jointly.  The function consumes only
    song-level metadata, so it can be run before audio is cut into clips.
    """

    rows = list(rows)
    if not rows:
        raise ValueError("at least one track row is required")
    if not fractions or any(value <= 0 for value in fractions.values()):
        raise ValueError("split fractions must be positive")
    total_fraction = float(sum(fractions.values()))
    fractions = {name: float(value) / total_fraction for name, value in fractions.items()}
    if len({str(row.get("artist", "")).strip() for row in rows}) < len(fractions):
        raise ValueError("not enough artists to populate every split")

    by_artist = defaultdict(list)
    for row in rows:
        artist = str(row.get("artist", "")).strip()
        if not artist:
            raise ValueError("every track row must have an artist")
        by_artist[artist].append(row)

    class_totals = Counter()
    artist_vectors = {}
    for artist, artist_rows in by_artist.items():
        counts = Counter()
        for row in artist_rows:
            annotated = str(row.get("activation_confidence_v2", "")).lower() in {
                "true", "1", "yes"
            }
            if annotated_only_for_balance and not annotated:
                continue
            counts.update(set(_families(row)).intersection(target_families))
        artist_vectors[artist] = counts
        class_totals.update(counts)

    # Rare and multi-class artists are placed first. The hash resolves ties
    # without relying on input or filesystem order.
    def artist_order(artist: str):
        rarity = sum(
            artist_vectors[artist][family] / max(class_totals[family], 1)
            for family in target_families
        )
        tie = hashlib.sha256((str(seed) + ":" + artist).encode("utf-8")).hexdigest()
        return (-rarity, -len(by_artist[artist]), tie)

    assignments: Dict[str, str] = {}
    split_tracks = Counter()
    split_classes: MutableMapping[str, Counter] = defaultdict(Counter)
    total_tracks = len(rows)

    def score(candidate: str, artist: str) -> float:
        value = 0.0
        for split_name, fraction in fractions.items():
            tracks = split_tracks[split_name]
            if split_name == candidate:
                tracks += len(by_artist[artist])
            track_target = max(total_tracks * fraction, 1.0)
            value += ((tracks - track_target) / track_target) ** 2
            for family in target_families:
                count = split_classes[split_name][family]
                if split_name == candidate:
                    count += artist_vectors[artist][family]
                target = class_totals[family] * fraction
                if target > 0:
                    value += ((count - target) / target) ** 2
        return value

    split_names = tuple(fractions)
    for artist in sorted(by_artist, key=artist_order):
        chosen = min(split_names, key=lambda name: (score(name, artist), name))
        assignments[artist] = chosen
        split_tracks[chosen] += len(by_artist[artist])
        split_classes[chosen].update(artist_vectors[artist])

    # Repair empty splits, which can occur only in very small synthetic audits.
    for empty in [name for name in split_names if name not in assignments.values()]:
        donor = max(split_names, key=lambda name: sum(v == name for v in assignments.values()))
        movable = sorted(
            (artist for artist, split in assignments.items() if split == donor),
            key=lambda artist: (len(by_artist[artist]), artist),
        )
        assignments[movable[0]] = empty

    return assignments


def assert_artist_disjoint(rows: Iterable[Mapping[str, object]]) -> None:
    """Raise if an artist occurs in more than one ``split`` value."""

    seen = {}
    for row in rows:
        artist = str(row["artist"])
        split = str(row["split"])
        previous = seen.setdefault(artist, split)
        if previous != split:
            raise ValueError("artist appears in multiple splits: " + artist)
