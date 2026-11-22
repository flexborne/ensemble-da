"""Synthetic analysis problems shared by the verification and timing scripts.

State: n components on [0, 1) with a smooth Gaussian-random-field ensemble (correlation length
0.05); p observations of distinct random components; Gaussian observation errors. Everything is
seeded, so both algorithms always see identical inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ensemble_da import Localization, estimate_covariance

CORR_LENGTH = 0.05
LOCALIZATION_RADIUS = 0.1


@dataclass
class Problem:
    Xf: np.ndarray          # (n, M) forecast ensemble
    H: np.ndarray           # (p,) observed components
    Y: np.ndarray           # (p, M) perturbed observations
    R: np.ndarray           # (p, p) observation-error covariance
    localization: Localization | None


def _random_fields(n, size, rng):
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    amplitude = np.exp(-0.25 * (k * CORR_LENGTH) ** 2)
    g = np.real(np.fft.ifft(np.fft.fft(rng.standard_normal((n, size)), axis=0) * amplitude[:, None], axis=0))
    return g / g.std()


def make_problem(n, p, M, seed, *, localized=True, correlated_R=False, obs_variance=0.01) -> Problem:
    rng = np.random.default_rng(seed)
    fields = _random_fields(n, M + 1, rng)
    truth, Xf = fields[:, 0], 0.5 * fields[:, 1:]
    H = np.sort(rng.choice(n, size=p, replace=False))
    if correlated_R:
        A = rng.standard_normal((p, p))
        R = obs_variance * ((A @ A.T) / p + np.eye(p))
    else:
        R = obs_variance * np.eye(p)
    y = truth[H] + np.linalg.cholesky(R) @ rng.standard_normal(p)
    Y = y[:, None] + np.linalg.cholesky(R) @ rng.standard_normal((p, M))
    coords = ((np.arange(n) + 0.5) / n)[:, None]
    loc = Localization(horizontal=coords, horizontal_radius=LOCALIZATION_RADIUS) if localized else None
    return Problem(Xf=Xf, H=H, Y=Y, R=R, localization=loc)


def background_covariance(problem: Problem) -> np.ndarray:
    B = estimate_covariance(problem.Xf)
    return B if problem.localization is None else B * problem.localization.matrix()
