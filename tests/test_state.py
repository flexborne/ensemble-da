"""Reduced multi-time window state."""
from __future__ import annotations

import numpy as np
import pytest

from ensemble_da import analysis_update, build_window_state, estimate_covariance


def test_single_time_window_is_the_filter_state():
    ens = np.arange(12.0).reshape(4, 3)
    ws = build_window_state([ens], [np.array([2, 0])])
    assert np.array_equal(ws.forecast, ens) and np.array_equal(ws.obs_rows, [2, 0])
    assert ws.target == slice(0, 4)


def test_earlier_times_keep_one_row_per_observed_component():
    rng = np.random.default_rng(0)
    e0, e1 = rng.standard_normal((5, 3)), rng.standard_normal((5, 3))
    observed = [np.array([3, 1, 3]), np.array([4])]           # component 3 observed twice at time 0
    ws = build_window_state([e0, e1], observed)
    assert np.array_equal(ws.component, [1, 3, 0, 1, 2, 3, 4])
    assert np.array_equal(ws.time_index, [0, 0, 1, 1, 1, 1, 1])
    assert np.array_equal(ws.forecast[ws.obs_rows], np.vstack([e0[[3, 1, 3]], e1[[4]]]))
    assert np.array_equal(ws.forecast[ws.target], e1)


def test_invalid_windows_are_rejected():
    with pytest.raises(ValueError, match='out of range'):
        build_window_state([np.zeros((3, 2))], [np.array([3])])
    with pytest.raises(ValueError, match='shape'):
        build_window_state([np.zeros((3, 2)), np.zeros((4, 2))], [np.array([0]), np.array([0])])


def test_reduced_window_equals_the_full_multi_time_analysis():
    """Keeping only the observed components of earlier times loses nothing at the target time."""
    rng = np.random.default_rng(1)
    n, M, K = 15, 30, 3
    joint = np.cumsum(rng.standard_normal((K * n, M)), axis=0)       # members correlated across times
    ensembles = [joint[k * n:(k + 1) * n] for k in range(K)]
    observed = [rng.choice(n, 4, replace=False) for _ in range(K)]
    p = sum(o.size for o in observed)
    R, Y = 0.3 * np.eye(p), rng.standard_normal((p, M))

    full_rows = np.concatenate([k * n + o for k, o in enumerate(observed)])
    full = analysis_update(estimate_covariance(joint), full_rows, Y, R, joint)[(K - 1) * n:]

    ws = build_window_state(ensembles, observed)
    reduced = analysis_update(estimate_covariance(ws.forecast), ws.obs_rows, Y, R, ws.forecast)[ws.target]
    assert np.allclose(reduced, full, atol=1e-10)
