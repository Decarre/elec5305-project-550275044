"""Canonical MedleyDB instrument-family mappings used by every pipeline stage."""

from typing import Dict, Iterable, Mapping, Optional, Sequence, Set


FAMILY_MEMBERS: Dict[str, Set[str]] = {
    "drums": {
        "drum set", "drum machine", "kick drum", "snare drum", "bass drum",
        "toms", "timpani", "bongo", "conga", "tambourine", "tabla",
        "auxiliary percussion",
    },
    "bass": {"electric bass", "double bass", "synth bass"},
    "guitar": {
        "acoustic guitar", "clean electric guitar", "distorted electric guitar",
        "slide guitar", "lap steel guitar",
    },
    "piano": {"piano", "tack piano", "electric piano"},
    "strings": {
        "violin", "viola", "cello", "violin section", "viola section",
        "cello section", "string section",
    },
    "vocals": {
        "male singer", "female singer", "male rapper", "female rapper",
        "male screamer", "female screamer", "vocalists", "choir",
    },
}

DEFAULT_TARGET_FAMILIES = ("drums", "bass", "guitar", "piano", "strings")


def normalize_instrument(name: str) -> str:
    """Return the normalized spelling used for taxonomy lookup."""

    return " ".join(str(name).strip().lower().replace("_", " ").split())


def validate_families(families: Sequence[str]) -> None:
    unknown = sorted(set(families).difference(FAMILY_MEMBERS))
    if unknown:
        raise ValueError("unknown instrument families: " + ", ".join(unknown))
    if len(set(families)) != len(families):
        raise ValueError("instrument family names must be unique")


def family_for_instrument(
    instrument: str, families: Optional[Sequence[str]] = None
) -> Optional[str]:
    """Map one MedleyDB instrument label to a selected broad family."""

    selected = tuple(families or FAMILY_MEMBERS)
    validate_families(selected)
    normalized = normalize_instrument(instrument)
    matches = [family for family in selected if normalized in FAMILY_MEMBERS[family]]
    if len(matches) > 1:
        raise ValueError("instrument maps to multiple families: " + normalized)
    return matches[0] if matches else None


def families_for_instruments(
    instruments: Iterable[str], families: Optional[Sequence[str]] = None
) -> Set[str]:
    """Return all selected families represented by an iterable of instruments."""

    result = set()
    for instrument in instruments:
        family = family_for_instrument(instrument, families)
        if family is not None:
            result.add(family)
    return result


def stem_family_map(
    stems: Mapping[str, Mapping[str, object]],
    families: Optional[Sequence[str]] = None,
) -> Dict[str, str]:
    """Return ``stem_id -> family`` for recognized, non-main-system stems."""

    result = {}
    for stem_id, metadata in stems.items():
        family = family_for_instrument(str(metadata.get("instrument", "")), families)
        if family is not None:
            result[str(stem_id)] = family
    return result
