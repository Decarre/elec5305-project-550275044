import numpy as np

from instrument_localization.postprocess import probabilities_to_intervals


def test_probabilities_are_converted_to_activity_intervals():
    probabilities = np.array(
        [
            [0.1, 0.8],
            [0.7, 0.9],
            [0.9, 0.2],
            [0.1, 0.1],
        ],
        dtype=np.float32,
    )

    intervals = probabilities_to_intervals(
        probabilities,
        ["guitar", "drums"],
        frame_hop_seconds=0.5,
        threshold=0.5,
    )

    assert [(item.instrument, item.start_time, item.end_time) for item in intervals] == [
        ("guitar", 0.5, 1.5),
        ("drums", 0.0, 1.0),
    ]

