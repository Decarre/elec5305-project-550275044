import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyse_track_errors import conditional_rates


def test_conditional_false_positives_have_the_right_denominators():
    truth=np.array([[0,1],[0,1],[0,0],[1,1]])
    predicted=np.array([[1,1],[0,1],[0,0],[1,1]])
    result=conditional_rates(truth,predicted,['a','b'])
    row=result[0]
    assert row['other_active_support']==2
    assert row['other_absent_support']==1
    assert row['fpr_other_active']==.5
    assert row['fpr_other_absent']==0


def test_missing_condition_is_not_zero_false_positive_rate():
    result=conditional_rates(np.array([[0,1],[0,1]]),np.ones((2,2)),['a','b'])
    assert result[0]['other_absent_support']==0
    assert result[0]['fpr_other_absent'] is None
