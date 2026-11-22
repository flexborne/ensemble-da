"""The high-level assimilate / assimilate_window interface."""
from __future__ import annotations

import numpy as np
import pytest

from ensemble_da import (Localization, Observations, analysis_update, assimilate, assimilate_window,
                         estimate_covariance, observation_covariance)


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    n, M = 30, 12
    Xf = 2.0 + np.cumsum(rng.standard_normal((n, M)), axis=0) / 10.0
    H = np.arange(1, n, 3)
    y = Xf.mean(axis=1)[H] + 0.3
    R = observation_covariance(0.1, n_obs=H.size)
    rho = Localization(horizontal=np.linspace(0, 1, n)[:, None], horizontal_radius=0.2).matrix()
    return Xf, y, H, R, rho


def test_deterministic_step_is_the_localized_kalman_update(problem):
    Xf, y, H, R, rho = problem
    res = assimilate(Xf, y, H, R, localization=rho, perturb=False)
    B = estimate_covariance(Xf) * rho
    assert np.allclose(res.analysis, analysis_update(B, H, np.repeat(y[:, None], Xf.shape[1], 1), R, Xf))


def test_seed_and_generator_give_identical_perturbations(problem):
    Xf, y, H, R, _ = problem
    a = assimilate(Xf, y, H, R, rng=7).analysis
    b = assimilate(Xf, y, H, R, rng=np.random.default_rng(7)).analysis
    assert np.array_equal(a, b) and not np.allclose(a, assimilate(Xf, y, H, R, rng=8).analysis)


def test_log_transform_analyses_log_values_and_returns_positive_states(problem):
    Xf, y, H, R, rho = problem
    res = assimilate(Xf, y, H, R, localization=rho, transform='log', perturb=False)
    Zf = np.log(Xf)
    expected = analysis_update(estimate_covariance(Zf) * rho, H, np.repeat(np.log(y)[:, None], Xf.shape[1], 1), R, Zf)
    assert np.all(res.analysis > 0) and np.allclose(np.log(res.analysis), expected)


def test_log_transform_rejects_non_positive_values(problem):
    Xf, y, H, R, _ = problem
    with pytest.raises(ValueError, match='positive'):
        assimilate(Xf, -y, H, R, transform='log')


def test_no_observations_leaves_the_forecast(problem):
    Xf = problem[0]
    res = assimilate(Xf, np.empty(0), np.empty(0, dtype=int), np.empty((0, 0)))
    assert np.array_equal(res.analysis, Xf) and res.diagnostics['n_obs'] == 0


def test_diagnostics_show_the_analysis_fits_the_observations_better(problem):
    Xf, y, H, R, rho = problem
    d = assimilate(Xf, y, H, R, localization=rho, perturb=False).diagnostics
    assert d['residual_rms'] < d['innovation_rms'] and d['spread_analysis'] < d['spread_forecast']
    assert (d['n_state'], d['n_obs'], d['n_members']) == (30, 10, 12)


@pytest.mark.parametrize('change, match', [
    (lambda a: {**a, 'forecast_ensemble': a['forecast_ensemble'][:, :1]}, 'M >= 2'),
    (lambda a: {**a, 'observation_operator': a['observation_operator'] + 100}, 'indices'),
    (lambda a: {**a, 'observation_operator': a['observation_operator'].astype(float)}, 'indices'),
    (lambda a: {**a, 'observation_covariance': np.eye(3)}, 'covariance'),
    (lambda a: {**a, 'localization': np.eye(3)}, 'localization'),
    (lambda a: {**a, 'transform': 'sqrt'}, 'transform'),
])
def test_invalid_inputs_are_rejected(problem, change, match):
    Xf, y, H, R, _ = problem
    args = {'forecast_ensemble': Xf, 'observations': y, 'observation_operator': H, 'observation_covariance': R}
    with pytest.raises(ValueError, match=match):
        assimilate(**change(args))


def test_one_time_window_equals_a_filter_step(problem):
    Xf, y, H, R, _ = problem
    loc = Localization(horizontal=np.linspace(0, 1, Xf.shape[0])[:, None], horizontal_radius=0.2)
    win = assimilate_window([Xf], [Observations(y, H, R)], localization=loc, perturb=False)
    single = assimilate(Xf, y, H, R, localization=loc.matrix(), perturb=False)
    assert np.allclose(win.analysis, single.analysis)


def test_window_uses_observations_from_other_times(problem):
    Xf, y, H, R, _ = problem
    earlier = Xf + 0.05
    with_past = assimilate_window([earlier, Xf], [Observations(y - 0.5, H, R), Observations.empty()], perturb=False)
    assert not np.allclose(with_past.analysis, Xf)            # the target moved using only past data
    with pytest.raises(TypeError):
        assimilate_window([Xf], [(y, H, R)])
