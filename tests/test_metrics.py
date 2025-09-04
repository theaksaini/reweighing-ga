import pandas as pd
import pytest

from utils import subgroup_FNR_loss

def test_subgroup_FNR_loss_expected_value():
    sample_df = pd.DataFrame([
        {'a': 1, 'b': 0, 'c': 1, 'y': 1},
        {'a': 1, 'b': 1, 'c': 1, 'y': 1},
        {'a': 0, 'b': 1, 'c': 1, 'y': 1},
        {'a': 0, 'b': 0, 'c': 0, 'y': 1},
        {'a': 1, 'b': 1, 'c': 0, 'y': 1},
        {'a': 1, 'b': 0, 'c': 1, 'y': 1},
        {'a': 1, 'b': 0, 'c': 1, 'y': 0},
        {'a': 0, 'b': 1, 'c': 1, 'y': 0},
        {'a': 0, 'b': 0, 'c': 0, 'y': 1},
        {'a': 1, 'b': 1, 'c': 1, 'y': 0},
    ], columns=['a', 'b', 'c', 'y'])

    # First three columns as features, last as target
    X = sample_df.iloc[:, :-1]
    y = sample_df.iloc[:, -1]
    y_pred = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    result = subgroup_FNR_loss(X, y, y_pred, sens_features=['a', 'b'])
    assert result == pytest.approx(0.2142857142857143, rel=1e-12)
