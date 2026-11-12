import numpy as np
from instrument_localization.alignment import aggregate_time_grid
from instrument_localization.metrics import multilabel_metrics, select_balanced_thresholds


def test_positive_predictor_cannot_hide_absence_errors():
    result = multilabel_metrics(np.array([[1], [1], [1], [0]]), np.ones((4, 1)))
    assert result['micro']['f1'] > .85
    assert result['micro']['balanced_accuracy'] == .5
    assert result['micro']['false_positive_rate'] == 1
    assert result['micro']['negative_f1'] == 0


def test_undefined_specificity_is_not_perfect_specificity():
    result = multilabel_metrics(np.ones((3, 1)), np.ones((3, 1)))
    assert result['micro']['specificity'] is None
    assert result['macro']['balanced_accuracy'] is None


def test_timestamp_boundaries_and_endpoint():
    values = np.arange(251, dtype=float)[None, :, None]
    result = aggregate_time_grid(values)
    assert result[0, 0, 0] == 12
    assert result[0, -1, 0] == 237
    mass = aggregate_time_grid(np.ones_like(values)/251, mass=True)
    assert np.allclose(mass, .1)
    assert np.isclose(mass.sum(), 1)


def test_balanced_threshold_uses_negatives():
    y = np.array([[1], [1], [0], [0]])
    score = np.array([[.8], [.7], [.3], [.2]])
    threshold = select_balanced_thresholds(y, score)
    assert multilabel_metrics(y, score, threshold)['micro']['balanced_accuracy'] == 1


def test_event_matching_penalises_missing_and_extra_events():
    from instrument_localization.events import event_metrics
    truth=np.array([[[0],[1],[1],[0],[0],[1],[0],[0]]])
    scores=np.array([[[0],[1],[1],[0],[0],[0],[0],[1]]])
    result=event_metrics(truth,scores,[.5],tolerance=.25)
    assert (result['tp'],result['fp'],result['fn'])==(1,1,1)
    assert result['event_f1']==.5


def test_interventions_supply_absence_and_known_switches():
    from pathlib import Path
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from prepare_controlled_babyslakh import control_gates
    absence=control_gates('absence',np.random.RandomState(5305))
    assert (absence.sum(axis=0)==0).sum()==2
    transitions=control_gates('transition',np.random.RandomState(5305))
    assert np.all(np.count_nonzero(np.diff(transitions,axis=0),axis=0)==1)
