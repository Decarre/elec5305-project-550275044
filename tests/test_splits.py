from instrument_localization.splits import assert_artist_disjoint, balanced_artist_split


def test_balanced_split_is_deterministic_and_artist_disjoint():
    rows = []
    for index in range(12):
        rows.append({
            "track_id": "track_" + str(index),
            "artist": "artist_" + str(index // 2),
            "families": "drums;guitar" if index % 2 else "bass;piano",
            "activation_confidence_v2": True,
        })
    first = balanced_artist_split(rows, ["drums", "bass", "guitar", "piano"])
    second = balanced_artist_split(rows, ["drums", "bass", "guitar", "piano"])
    assert first == second
    assigned = [{**row, "split": first[row["artist"]]} for row in rows]
    assert set(first.values()) == {"train", "validation", "test"}
    assert_artist_disjoint(assigned)
