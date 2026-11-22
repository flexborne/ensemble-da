"""Observation-error covariance, perturbed observations and observation records."""
from __future__ import annotations

import numpy as np
import pytest

from ensemble_da import Observations, observation_covariance, perturbed_observations


def test_measurement_and_representativeness_variances_add():
    R = observation_covariance([0.1, 0.2], representativeness_std=[0.3, 0.0])
    assert np.allclose(R, np.diag([0.01 + 0.09, 0.04]))
    assert np.allclose(observation_covariance(0.1, 0.2, n_obs=3), 0.05 * np.eye(3))


@pytest.mark.parametrize('std', [0.0, -0.1, float('nan')])
def test_non_positive_or_non_finite_errors_are_rejected(std):
    with pytest.raises(ValueError):
        observation_covariance(std, n_obs=2)


def test_diagonal_perturbations_are_scaled_standard_normals():
    y = np.array([1.0, 2.0, 3.0])
    Y = perturbed_observations(y, 0.04 * np.eye(3), 5, np.random.default_rng(0))
    assert np.allclose(Y, y[:, None] + 0.2 * np.random.default_rng(0).standard_normal((3, 5)))


def test_correlated_perturbations_have_covariance_R():
    R = np.array([[1.0, 0.6], [0.6, 2.0]])
    Y = perturbed_observations(np.zeros(2), R, 200_000, np.random.default_rng(1))
    assert np.allclose(np.cov(Y), R, atol=0.02)


def test_observation_record_validates_shapes():
    Observations([1.0, 2.0], [0, 3], np.eye(2))
    with pytest.raises(ValueError, match='indices'):
        Observations([1.0, 2.0], [0], np.eye(2))
    with pytest.raises(ValueError, match='covariance'):
        Observations([1.0, 2.0], [0, 1], np.eye(3))
    assert Observations.empty().values.size == 0
