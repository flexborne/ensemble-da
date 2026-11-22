"""Model-independent sequential cycle: planning and execution with an injected forecast."""
from __future__ import annotations

import numpy as np
import pytest

from ensemble_da import (Localization, Observations, assimilate_window, derive_windows, observation_covariance,
                         plan_cycle, run_cycle)


def _kinds(plan):
    return [(s.kind, s.time) for s in plan]


def test_derive_windows():
    first, last = derive_windows([0.0, 0.75, 1.5, 1.75, 1.875, 2.0, 2.1, 4.0], 0.25)
    assert first == [0, 1, 2, 2, 3, 3, 4, 7]
    assert last == [0, 1, 3, 5, 6, 6, 6, 7]


def test_derive_windows_with_large_time_values():
    times = 1.7e9 + 0.1 * np.arange(5)                    # e.g. epoch seconds with sub-second steps
    assert derive_windows(times, 0.1) == ([0, 0, 1, 2, 3], [1, 2, 3, 4, 4])


def test_filter_and_noda_plans():
    assert _kinds(plan_cycle('filter', [1, 2, 3])) == [('analyze', 1), ('forecast', 2), ('analyze', 2), ('forecast', 3)]
    noda = plan_cycle('noda', [1, 2, 3])
    assert _kinds(noda) == [('forecast', 2), ('forecast', 3)] and {s.restart for s in noda} == {'forecast'}


def test_smoother_plan_windows_and_lookahead():
    plan = plan_cycle('smoother', [0.0, 0.25, 0.375], window=0.25)
    assert _kinds(plan) == [('forecast', 0.25),                       # free run to the first window's end
                            ('analyze', 0.0), ('forecast', 0.25), ('forecast', 0.375),
                            ('analyze', 0.25), ('forecast', 0.375)]
    assert plan[1].window == (0.25, 0.0)                              # neighbours, then the target
    assert plan[4].window == (0.0, 0.375, 0.25)
    assert [s.restart for s in plan if s.kind == 'forecast'] == ['forecast', 'analysis', 'forecast', 'analysis']


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match='mode'):
        plan_cycle('kalman', [0, 1])


@pytest.fixture
def linear_case():
    rng = np.random.default_rng(0)
    n, M = 8, 10
    times = [0.0, 1.0, 2.0, 3.0]
    initial = rng.standard_normal((n, M))
    truth = np.ones(n)
    H = np.array([1, 4, 6])
    R = observation_covariance(0.1, n_obs=3)
    obs = {t: Observations(truth[H] + 0.05 * t, H, R) for t in times}
    forecast = lambda ens, t0, t1: 0.9 * ens + 0.1 * (t1 - t0)       # noqa: E731  (a simple linear model)
    return forecast, initial, times, obs


def test_noda_never_analyses(linear_case):
    forecast, initial, times, obs = linear_case
    out = run_cycle(forecast, initial, times, obs, mode='noda')
    assert out.analyses == {} and set(out.forecasts) == set(times)
    assert np.allclose(out.forecasts[3.0], forecast(forecast(forecast(initial, 0, 1), 1, 2), 2, 3))


def test_filter_analyses_every_time_but_the_last(linear_case):
    forecast, initial, times, obs = linear_case
    out = run_cycle(forecast, initial, times, obs, mode='filter', rng=1)
    assert set(out.analyses) == {0.0, 1.0, 2.0}
    assert np.allclose(out.forecasts[1.0], forecast(out.analyses[0.0], 0.0, 1.0))
    assert [s['id'] for s in out.steps] == list(range(len(out.steps)))
    step = next(s for s in out.steps if s['step'] == 'forecast' and s['time'] == 3.0)
    o = obs[3.0]
    expected = np.sqrt(np.mean((o.values - out.forecasts[3.0].mean(axis=1)[o.indices]) ** 2))
    assert step['innovation_rms'] == pytest.approx(expected)


def test_filter_cycle_equals_manual_steps_with_one_generator():
    """transform, localization (with decoupled components) and the shared generator reach every step."""
    rng = np.random.default_rng(2)
    n, M, times = 10, 8, [0.0, 1.0, 2.0]
    initial = np.exp(0.3 * rng.standard_normal((n, M)))
    H = np.array([0, 3, 7])
    obs = {t: Observations(np.exp(rng.standard_normal(3)), H, observation_covariance(0.2, n_obs=3)) for t in times}
    loc = Localization(horizontal=np.arange(n)[:, None], horizontal_radius=3.0, decoupled=np.arange(n) == 5)
    forecast = lambda ens, t0, t1: ens * 1.1                            # noqa: E731
    out = run_cycle(forecast, initial, times, obs, mode='filter', localization=loc, transform='log', rng=5)

    g, state = np.random.default_rng(5), initial
    for t, nxt in zip(times, times[1:]):
        analysis = assimilate_window([state], [obs[t]], [t], localization=loc, transform='log', rng=g).analysis
        assert np.array_equal(out.analyses[t], analysis)
        state = forecast(analysis, t, nxt)


def test_a_model_that_works_in_place_cannot_corrupt_stored_states(linear_case):
    forecast, initial, times, obs = linear_case
    def in_place(ens, t0, t1):
        ens *= 0.9
        ens += 0.1 * (t1 - t0)
        return ens
    keep = initial.copy()
    a = run_cycle(in_place, initial, times, obs, mode='smoother', window=1.0, perturb=False)
    b = run_cycle(forecast, initial, times, obs, mode='smoother', window=1.0, perturb=False)
    assert np.array_equal(initial, keep)
    assert all(np.allclose(a.analyses[t], b.analyses[t]) for t in b.analyses)


def test_cycle_is_reproducible_with_a_seed(linear_case):
    forecast, initial, times, obs = linear_case
    a = run_cycle(forecast, initial, times, obs, mode='filter', rng=3)
    b = run_cycle(forecast, initial, times, obs, mode='filter', rng=3)
    assert all(np.array_equal(a.analyses[t], b.analyses[t]) for t in a.analyses)


def test_smoother_uses_future_observations(linear_case):
    forecast, initial, times, obs = linear_case
    filt = run_cycle(forecast, initial, times, obs, mode='filter', perturb=False)
    smo = run_cycle(forecast, initial, times, obs, mode='smoother', window=1.0, perturb=False)
    zero = run_cycle(forecast, initial, times, obs, mode='smoother', window=0.0, perturb=False)
    assert not np.allclose(smo.analyses[0.0], filt.analyses[0.0])
    assert all(np.allclose(zero.analyses[t], filt.analyses[t]) for t in filt.analyses)


def test_invalid_cycles_are_rejected(linear_case):
    forecast, initial, times, obs = linear_case
    with pytest.raises(ValueError, match='increasing'):
        run_cycle(forecast, initial, [0.0, 0.0], {})
    with pytest.raises(ValueError, match='outside'):
        run_cycle(forecast, initial, times, {**obs, 9.0: obs[0.0]})
    with pytest.raises(ValueError, match='shape'):
        run_cycle(lambda e, a, b: e[:2], initial, times, obs, mode='noda')
