from instrument_localization.slakh import (
    assert_group_disjoint,
    grouped_track_split,
    normalise_inst_class,
)


def test_slakh_class_mapping():
    assert normalise_inst_class("Drums") == "drums"
    assert normalise_inst_class("Piano") == "piano"
    assert normalise_inst_class("Strings") == ""


def test_duplicate_groups_never_cross_partitions():
    groups = {"a": "same", "b": "same", "c": "c", "d": "d", "e": "e"}
    split = grouped_track_split(groups, seed=7)
    assert split["a"] == split["b"]
    assert_group_disjoint(groups, split)
