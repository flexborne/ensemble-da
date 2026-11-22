"""Observations: error covariance R, perturbed observation ensembles, per-time observation records."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def observation_covariance(measurement_std, representativeness_std=0.0, n_obs=None) -> np.ndarray:
    """Diagonal observation-error covariance R = diag(sigma_meas^2 + sigma_repr^2).

    Measurement and representativeness errors are independent contributions given separately,
    per observation (arrays of length p) or as scalars broadcast to `n_obs` observations.
    For a log-transformed analysis the standard deviations are in log space (e.g. ln GSD).
    """
    m = np.asarray(measurement_std, dtype=float)
    r = np.asarray(representativeness_std, dtype=float)
    if not (np.all(np.isfinite(m) & (m >= 0)) and np.all(np.isfinite(r) & (r >= 0))):
        raise ValueError('error standard deviations must be finite and non-negative')
    var = m ** 2 + r ** 2
    var = np.broadcast_to(var, (n_obs,)) if n_obs is not None else np.atleast_1d(var)
    if np.any(var <= 0):
        raise ValueError('every observation needs a positive total error variance')
    return np.diag(var.astype(float))


def perturbed_observations(y, R, n_members, rng) -> np.ndarray:
    """Observation ensemble (p, M): y plus N(0, R) perturbations (stochastic EnKF)."""
    y = np.asarray(y, dtype=float).ravel()
    L = np.linalg.cholesky(np.asarray(R, dtype=float))
    return y[:, None] + L @ rng.standard_normal((y.size, n_members))


@dataclass(frozen=True, eq=False)
class Observations:
    """Observations at one time: values y (p,), observed state indices (p,), error covariance R (p, p)."""

    values: np.ndarray
    indices: np.ndarray
    covariance: np.ndarray

    def __post_init__(self):
        values = np.asarray(self.values, dtype=float).ravel()
        indices = np.asarray(self.indices, dtype=int).ravel()
        cov = np.asarray(self.covariance, dtype=float)
        if indices.size != values.size:
            raise ValueError(f'{indices.size} indices for {values.size} observation values')
        if cov.shape != (values.size, values.size):
            raise ValueError(f'covariance must be ({values.size}, {values.size}), got {cov.shape}')
        object.__setattr__(self, 'values', values)
        object.__setattr__(self, 'indices', indices)
        object.__setattr__(self, 'covariance', cov)

    @classmethod
    def empty(cls) -> 'Observations':
        return cls(np.empty(0), np.empty(0, dtype=int), np.empty((0, 0)))
