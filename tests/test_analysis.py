"""Algebraic properties of the observation-space Kalman update."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import random_problem

from ensemble_da import analysis_update, estimate_covariance


def test_scalar_gaussian_update_matches_closed_form():
    B = np.array([[4.0, 1.0], [1.0, 2.0]])
    H = np.array([0])
    R = np.array([[0.5]])
    Xf = np.array([[1.0], [2.0]])
    Y = np.array([[3.0]])
    K = B[:, H] @ np.linalg.inv(B[np.ix_(H, H)] + R)
    assert np.allclose(analysis_update(B, H, Y, R, Xf), Xf + K @ (Y - Xf[H]), atol=1e-12)


def test_uncorrelated_unobserved_component_is_unchanged():
    B = np.array([[3.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 5.0]])
    Xa = analysis_update(B, np.array([0, 1]), np.array([[2.0], [0.0]]), 0.3 * np.eye(2),
                         np.array([[1.0], [1.0], [7.0]]))
    assert Xa[2, 0] == pytest.approx(7.0, abs=1e-12)


def test_unobserved_components_do_not_change_the_observed_update():
    # marginalization: dropping components that are neither observed nor wanted is exact
    B, H, Y, R, Xf = random_problem(12, 3, 5, seed=2, localized=True)
    wanted = np.array(sorted(set(H) | {0, 11}))
    pos = {c: i for i, c in enumerate(wanted)}
    full = analysis_update(B, H, Y, R, Xf)
    reduced = analysis_update(B[np.ix_(wanted, wanted)], [pos[c] for c in H], Y, R, Xf[wanted])
    assert np.allclose(full[wanted], reduced, atol=1e-10)


def test_index_and_matrix_operators_agree():
    B, H, Y, R, Xf = random_problem(20, 6, 8, seed=3, localized=True)
    assert np.allclose(analysis_update(B, H, Y, R, Xf), analysis_update(B, np.eye(20)[H], Y, R, Xf), atol=1e-12)


def test_general_linear_operator_matches_explicit_gain():
    B, H, Y, R, Xf = random_problem(15, 4, 6, seed=4, localized=True, matrix_H=True)
    K = B @ H.T @ np.linalg.inv(H @ B @ H.T + R)
    assert np.allclose(analysis_update(B, H, Y, R, Xf), Xf + K @ (Y - H @ Xf), atol=1e-10)


def test_singular_covariance_keeps_increment_in_its_range():
    # unlocalized covariance of M = 4 members has rank 3: the increment must lie in range(B)
    rng = np.random.default_rng(5)
    Xf = rng.standard_normal((30, 4))
    B = estimate_covariance(Xf)
    increment = analysis_update(B, np.arange(0, 30, 3), rng.standard_normal((10, 4)), 0.2 * np.eye(10), Xf) - Xf
    anomalies = Xf - Xf.mean(axis=1, keepdims=True)
    coeffs = np.linalg.lstsq(anomalies, increment, rcond=None)[0]
    assert np.allclose(anomalies @ coeffs, increment, atol=1e-10)
