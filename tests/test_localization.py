"""Distance-based localization factors."""
from __future__ import annotations

import numpy as np
import pytest

from ensemble_da import Localization, great_circle_distance


def test_horizontal_factors_decay_exponentially():
    x = np.array([[0.0], [0.1], [0.3]])
    rho = Localization(horizontal=x, horizontal_radius=0.1).matrix()
    assert np.allclose(rho, rho.T) and np.allclose(np.diag(rho), 1.0)
    assert rho[0, 1] == pytest.approx(np.exp(-1.0)) and rho[0, 2] == pytest.approx(np.exp(-3.0))


def test_great_circle_distance_of_one_degree_of_latitude():
    d = great_circle_distance([[45.0, 30.0]], [[46.0, 30.0]])
    assert d[0, 0] == pytest.approx(2 * np.pi * 6_371_000.0 / 360.0, rel=1e-12)


def test_vertical_and_temporal_factors_multiply():
    loc = Localization(horizontal=np.zeros((2, 1)), vertical=np.array([0.0, 50.0]),
                       horizontal_radius=1.0, vertical_radius=100.0, time_radius=2.0)
    rho = loc.matrix(times=[0.0, 1.0])
    assert rho[0, 1] == pytest.approx(np.exp(-0.5) * np.exp(-0.5))


def test_window_rows_repeat_components_at_different_times():
    loc = Localization(horizontal=np.array([[0.0], [1.0]]), horizontal_radius=1.0, time_radius=0.5)
    rho = loc.matrix(components=[0, 0, 1], times=[0.0, 0.25, 0.25])
    assert rho[0, 1] == pytest.approx(np.exp(-0.5))           # same place, 0.25 apart in time
    assert rho[1, 2] == pytest.approx(np.exp(-1.0))           # same time, unit distance


def test_decoupled_components_keep_only_their_variance():
    loc = Localization(horizontal=np.zeros((3, 1)), horizontal_radius=1.0, time_radius=1.0,
                       decoupled=np.array([False, True, False]))
    rho = loc.matrix(components=[0, 1, 1, 2], times=[0, 0, 1, 1])
    assert np.allclose(rho[[1, 2], :][:, [0, 3]], 0.0)        # no covariance with other components
    assert rho[1, 2] == 0.0 and rho[1, 1] == rho[2, 2] == 1.0  # nor with itself at another time
    assert rho[0, 3] > 0


@pytest.mark.parametrize('kwargs', [{'metric': 'manhattan'}, {'horizontal_radius': 0.0},
                                    {'time_radius': float('nan')}])
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Localization(**kwargs)


def test_components_required_without_coordinates():
    with pytest.raises(ValueError, match='components'):
        Localization(time_radius=1.0).matrix()
