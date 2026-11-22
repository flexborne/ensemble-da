"""Shared test helpers. The toy forecast model lives in examples/, outside the package."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ensemble_da import Localization, estimate_covariance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'examples'))


def random_problem(n, p, M, seed, *, localized=False, correlated_R=False, matrix_H=False):
    """A random analysis problem (B, H, Y, R, Xf) with a smooth ensemble and optional extras."""
    rng = np.random.default_rng(seed)
    Xf = np.cumsum(rng.standard_normal((n, M)), axis=0) / np.sqrt(n)    # spatially correlated members
    B = estimate_covariance(Xf)
    if localized:
        x = np.linspace(0.0, 1.0, n)[:, None]
        B = B * Localization(horizontal=x, horizontal_radius=0.15).matrix()
    if correlated_R:
        A = rng.standard_normal((p, p))
        R = 0.05 * (A @ A.T) / p + 0.1 * np.eye(p)
    else:
        R = np.diag(rng.uniform(0.05, 0.5, p))
    H = rng.choice(n, size=p, replace=False)
    if matrix_H:
        H = np.eye(n)[H] * 0.5 + np.eye(n)[np.roll(H, 1)] * 0.5          # averages of two components
    Y = (H @ Xf if np.ndim(H) == 2 else Xf[H]) + rng.standard_normal((p, M))
    return B, H, Y, R, Xf
