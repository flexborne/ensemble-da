"""The production update against an independent state-space least-squares solver.

reference_analysis shares no code with analysis_update (eigendecomposition + SVD least squares
in state space vs a Cholesky factorization of the p x p innovation covariance), so agreement on
full ensembles verifies the implementation rather than re-running it.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import random_problem

from ensemble_da import analysis_update, reference_analysis

CASES = [
    # n,   p,  M,  localized, correlated R, matrix H
    (10, 1, 24, True, False, False),
    (10, 10, 3, False, False, False),      # M << n, every component observed
    (40, 8, 24, True, False, False),
    (40, 8, 24, False, True, False),
    (80, 16, 5, False, False, False),      # strongly degenerate: rank(B) = 4
    (80, 16, 5, True, True, False),
    (120, 30, 24, True, False, False),
    (120, 30, 60, False, False, False),
    (60, 12, 24, True, False, True),       # general linear observation operator
    (60, 12, 4, False, True, True),
]


@pytest.mark.parametrize('n,p,M,localized,correlated_R,matrix_H', CASES,
                         ids=[f'n{c[0]}-p{c[1]}-M{c[2]}' + '-loc' * c[3] + '-corrR' * c[4] + '-matH' * c[5] for c in CASES])
@pytest.mark.parametrize('seed', [0, 1])
def test_update_matches_independent_reference(n, p, M, localized, correlated_R, matrix_H, seed):
    B, H, Y, R, Xf = random_problem(n, p, M, seed, localized=localized, correlated_R=correlated_R, matrix_H=matrix_H)
    production = analysis_update(B, H, Y, R, Xf)
    reference = reference_analysis(B, H, Y, R, Xf)
    scale = np.abs(reference - Xf).max()
    assert np.abs(production - reference).max() <= 1e-9 * max(scale, 1.0)


def test_degenerate_ensemble_covariance_is_handled():
    B, H, Y, R, Xf = random_problem(100, 20, 6, seed=7)
    assert np.linalg.matrix_rank(B) == 5                    # M - 1, far below n
    assert np.allclose(analysis_update(B, H, Y, R, Xf), reference_analysis(B, H, Y, R, Xf), atol=1e-9)
