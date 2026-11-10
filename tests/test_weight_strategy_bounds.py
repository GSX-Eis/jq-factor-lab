"""Portfolio limits must hold after final normalization."""
import numpy as np
import pandas as pd
import pytest

from factor_synthesis.weighting import get_weight_strategy


@pytest.mark.parametrize('method', ['score', 'rank', 'softmax'])
@pytest.mark.parametrize('scores', [[100., 1., 0., -1., -2.], [1.] * 5, [10000., 0., 0., 0., 0.], [1e308, 1., 0., -1., -1e308]])
@pytest.mark.parametrize('minimum', [0., .05, .2])
def test_feasible_bounds_are_preserved(method, scores, minimum):
    values = pd.Series(scores, index=list('ABCDE'))
    weights = get_weight_strategy(method, dict(top_pct=1, min_weight=minimum, max_weight=.3)).calculate_weights(values)
    assert weights.sum() == pytest.approx(1)
    assert weights.max() <= .3 + 1e-12
    assert weights.min() >= minimum - 1e-12
    assert weights.index.is_unique
    assert set(weights.index) <= set(values.index)
    assert np.isfinite(weights).all()


@pytest.mark.parametrize('method', ['score', 'rank', 'softmax'])
@pytest.mark.parametrize('options', [dict(top_pct=.4, max_weight=.3), dict(min_weight=.21, max_weight=.4)])
def test_infeasible_limits_fail_instead_of_being_overridden(method, options):
    config = dict(top_pct=1, min_weight=0, max_weight=1)
    config.update(options)
    with pytest.raises(ValueError, match='Infeasible'):
        get_weight_strategy(method, config).calculate_weights(pd.Series([5., 4., 3., 2., 1.]))


@pytest.mark.parametrize('method', ['score', 'rank', 'softmax'])
def test_selection_and_empty_input(method):
    strategy = get_weight_strategy(method, dict(top_pct=.5, max_weight=.3, min_weight=.01))
    assert strategy.calculate_weights(pd.Series(dtype=float)).empty
    weights = strategy.calculate_weights(pd.Series(range(10)))
    assert set(weights.index) == set(range(5, 10))
    assert weights.sum() == pytest.approx(1)
