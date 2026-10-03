import numpy as np

from instrument_localization.annotations import (
    aggregate_stems_to_families,
    resample_activity_grid,
)


def test_family_aggregation_and_reporting_grid():
    curves = {
        "S01": np.array([0.0, 0.8, 0.2, 0.0]),
        "S02": np.array([0.1, 0.3, 0.9, 0.0]),
    }
    activity = aggregate_stems_to_families(
        curves, {"S01": "guitar", "S02": "guitar"}, ["guitar", "piano"]
    )
    assert np.allclose(activity[:, 0], [0.1, 0.8, 0.9, 0.0])
    centres, coarse = resample_activity_grid(
        np.array([0.0, 0.25, 0.5, 0.75]), activity, 0.0, 1.0, 0.5, "max"
    )
    assert np.allclose(centres, [0.25, 0.75])
    assert np.allclose(coarse[:, 0], [0.8, 0.9])
