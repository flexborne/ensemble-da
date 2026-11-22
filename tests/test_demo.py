"""The model-independent twin experiment reproduces its documented behaviour, over several seeds."""
from __future__ import annotations

import pytest

import demo_transport_1d as demo


@pytest.fixture(scope='module')
def summary():
    return demo.summarize(range(1, 11))


def test_one_analysis_cuts_the_forecast_error(summary):
    assert summary['analysis / forecast (one analysis)']['max'] < 0.5


def test_localization_beats_the_raw_ensemble_covariance(summary):
    s = summary['localized / unlocalized analysis']
    assert s['below_1'] == s['seeds'] and s['median'] < 0.7


def test_cycling_beats_the_free_run_and_analyses_beat_their_forecasts(summary):
    assert summary['filter / free-run forecast']['median'] < 0.3
    assert summary['filter analysis / its forecast']['median'] < 0.9


def test_smoother_beats_the_filter_on_average(summary):
    s = summary['smoother / filter analysis']
    assert s['median'] < 0.95 and s['below_1'] >= 8
