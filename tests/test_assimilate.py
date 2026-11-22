"""The high-level assimilate / assimilate_window interface."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import block_diag

from ensemble_da import (Localization, Observations, analysis_update, assimilate, assimilate_window,
                         estimate_covariance, observation_covariance, reference_analysis)
from ensemble_da.transform import from_log, to_log


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    n, M = 30, 12
    Xf = 2.0 + np.cumsum(rng.standard_normal((n, M)), axis=0) / 10.0
    H = np.arange(1, n, 3)
    y = Xf.mean(axis=1)[H] + 0.3
    R = observation_covariance(0.1, n_obs=H.size)
    loc = Localization(horizontal=np.linspace(0, 1, n)[:, None], horizontal_radius=0.2)
    return Xf, y, H, R, loc


def test_deterministic_step_is_the_localized_kalman_update(problem):
    Xf, y, H, R, loc = problem
    res = assimilate(Xf, y, H, R, localization=loc, perturb=False)
    B = estimate_covariance(Xf) * loc.matrix()
    assert np.allclose(res.analysis, analysis_update(B, H, np.repeat(y[:, None], Xf.shape[1], 1), R, Xf))
    assert np.allclose(assimilate(Xf, y, H, R, localization=loc.matrix(), perturb=False).analysis, res.analysis)


def test_seed_and_generator_give_identical_perturbations(problem):
    Xf, y, H, R, _ = problem
    a = assimilate(Xf, y, H, R, rng=7).analysis
    b = assimilate(Xf, y, H, R, rng=np.random.default_rng(7)).analysis
    assert np.array_equal(a, b) and not np.allclose(a, assimilate(Xf, y, H, R, rng=8).analysis)


def test_log_transform_works_in_log_space(problem):
    Xf, y, H, R, loc = problem
    res = assimilate(Xf, y, H, R, localization=loc, transform='log', perturb=False)
    Zf, z = np.log(Xf), np.log(y)
    expected = analysis_update(estimate_covariance(Zf) * loc.matrix(), H, np.repeat(z[:, None], Xf.shape[1], 1), R, Zf)
    assert np.allclose(np.log(res.analysis), expected)
    innovation = z - Zf.mean(axis=1)[H]                      # diagnostics are in the analysis (log) space
    assert res.diagnostics['innovation_rms'] == pytest.approx(np.sqrt(np.mean(innovation ** 2)))


def test_log_transform_guards():
    with pytest.raises(ValueError, match='positive'):
        to_log(np.array([1.0, 0.0]))
    with np.errstate(over='ignore'), pytest.raises(FloatingPointError):
        from_log(np.array([0.0, 800.0]))                      # overflow
    with pytest.raises(FloatingPointError):
        from_log(np.array([0.0, -800.0]))                     # underflow to zero is not positive


def test_no_observations_leaves_the_forecast(problem):
    Xf = problem[0]
    res = assimilate(Xf, np.empty(0), np.empty(0, dtype=int), np.empty((0, 0)))
    assert np.array_equal(res.analysis, Xf) and res.diagnostics['n_obs'] == 0


def test_diagnostics_show_the_analysis_fits_the_observations_better(problem):
    Xf, y, H, R, loc = problem
    d = assimilate(Xf, y, H, R, localization=loc, perturb=False).diagnostics
    assert d['residual_rms'] < d['innovation_rms'] and d['spread_analysis'] < d['spread_forecast']
    assert (d['n_state'], d['n_obs'], d['n_members']) == (30, 10, 12)


@pytest.mark.parametrize('change', [
    lambda a: {**a, 'forecast_ensemble': a['forecast_ensemble'][:, :1]},           # one member
    lambda a: {**a, 'observation_operator': a['observation_operator'] + 100},     # index out of range
    lambda a: {**a, 'observation_operator': -a['observation_operator']},          # negative index
    lambda a: {**a, 'observation_operator': a['observation_operator'] + 0.5},     # non-integer index
    lambda a: {**a, 'observation_covariance': np.full(10, 0.01)},                 # R as a vector
    lambda a: {**a, 'transform': 'sqrt'},
])
def test_inputs_that_would_give_silently_wrong_results_are_rejected(problem, change):
    Xf, y, H, R, _ = problem
    args = {'forecast_ensemble': Xf, 'observations': y, 'observation_operator': H, 'observation_covariance': R}
    with pytest.raises(ValueError):
        assimilate(**change(args), perturb=False)


def test_window_matches_the_reference_solution_of_the_stacked_problem():
    """K = 3 times, different R per time, temporal localization, log transform."""
    rng = np.random.default_rng(1)
    n, M, times = 12, 25, [0.0, 0.5, 1.0]
    joint = 1.0 + np.exp(0.2 * np.cumsum(rng.standard_normal((3 * n, M)), axis=0))
    ensembles = [joint[k * n:(k + 1) * n] for k in range(3)]
    obs = [Observations(rng.uniform(1, 3, 4), rng.choice(n, 4, replace=False), observation_covariance(s, n_obs=4))
           for s in (0.05, 0.1, 0.2)]
    loc = Localization(horizontal=np.linspace(0, 1, n)[:, None], horizontal_radius=0.3, time_radius=0.4)
    res = assimilate_window(ensembles, obs, times, localization=loc, transform='log', perturb=False)

    rows = np.concatenate([np.arange(k * n, (k + 1) * n) for k in range(3)])     # every component, every time
    component, row_time = rows % n, np.repeat(times, n)
    Z = np.log(joint)
    B = estimate_covariance(Z) * loc.matrix(component, row_time)
    H = np.concatenate([k * n + o.indices for k, o in enumerate(obs)])
    y = np.log(np.concatenate([o.values for o in obs]))
    ref = reference_analysis(B, H, np.repeat(y[:, None], M, 1), block_diag(*[o.covariance for o in obs]), Z)
    assert np.allclose(np.log(res.analysis), ref[2 * n:], atol=1e-9)
    assert res.diagnostics['n_state'] == n and res.diagnostics['window_times'] == 3
