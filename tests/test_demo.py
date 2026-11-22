"""The model-independent twin experiment reproduces its documented behaviour."""
from __future__ import annotations

import numpy as np
import pytest

import demo_transport_1d as demo


@pytest.fixture(scope='module')
def result():
    return demo.run(seed=1)


def _mean(d):
    return float(np.mean(list(d.values())))


def test_analysis_reduces_the_forecast_error(result):
    s = result['single_analysis_t1']
    assert s['analysis_rmse_localized'] < 0.25 * s['forecast_rmse']


def test_localization_beats_the_raw_ensemble_covariance(result):
    s = result['single_analysis_t1']
    assert s['analysis_rmse_localized'] < s['analysis_rmse_unlocalized']


def test_cycling_beats_the_free_run(result):
    c = result['cycle']
    assert _mean(c['filter']['forecast_rmse']) < 0.2 * _mean(c['noda']['forecast_rmse'])
    assert _mean(c['smoother']['forecast_rmse']) < _mean(c['filter']['forecast_rmse'])
    assert _mean(c['smoother']['analysis_rmse']) < _mean(c['filter']['analysis_rmse'])
