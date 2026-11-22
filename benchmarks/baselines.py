"""Benchmark baseline: the same analysis solved in state space with Cholesky factorizations."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve, cholesky


def state_space_cholesky(B, H, Y, R, Xf):
    """Information form in state space with B = L L^T and x = xf + L w:

        (I + (H L)^T R^-1 H L) w = (H L)^T R^-1 (y - H xf).

    The usual state-space (regression) formulation: an n x n Cholesky system, O(n^3). Needs a
    positive-definite B (true for a localized covariance); H holds observed state indices.
    """
    L = cholesky(B, lower=True)
    HL = L[H]
    R_factor = cho_factor(R, lower=True)
    A = np.eye(B.shape[0]) + HL.T @ cho_solve(R_factor, HL)
    W = cho_solve(cho_factor(A, lower=True), HL.T @ cho_solve(R_factor, Y - Xf[H]))
    return Xf + L @ W
