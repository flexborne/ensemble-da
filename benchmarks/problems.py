"""Synthetic analysis problems shared by the verification and timing scripts.

State: n components on [0, 1) with a smooth Gaussian-random-field ensemble (correlation length
0.05); p observations of distinct random components; Gaussian observation errors. A smoother
window holds K such states 0.5 apart in time, correlated across times. Everything is seeded, so
all solvers see identical inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import block_diag

from ensemble_da import Localization, Observations, build_window_state, estimate_covariance, perturbed_observations

CORR_LENGTH = 0.05
LOCALIZATION_RADIUS = 0.1
TIME_STEP, TIME_RADIUS = 0.5, 1.0          # smoother windows
TIME_CORRELATION = 0.8                     # x_k = 0.8 x_(k-1) + 0.6 w_k


@dataclass
class Problem:
    """One analysis problem; for a smoother window, its reduced window state."""
    Xf: np.ndarray          # (N, M) forecast ensemble
    H: np.ndarray           # (p,) observed rows
    Y: np.ndarray           # (p, M) perturbed observations
    R: np.ndarray           # (p, p) observation-error covariance
    localization: Localization | None
    rows: tuple = ()        # Localization.matrix(*rows): (component, time) of each window-state row


@dataclass
class WindowProblem:
    ensembles: list         # K forecast ensembles (n, M), the target time last
    observations: list      # K Observations
    times: np.ndarray       # (K,)
    localization: Localization | None


def _random_fields(n, size, rng):
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    amplitude = np.exp(-0.25 * (k * CORR_LENGTH) ** 2)
    g = np.real(np.fft.ifft(np.fft.fft(rng.standard_normal((n, size)), axis=0) * amplitude[:, None], axis=0))
    return g / g.std()


def _covariance(p, variance, correlated, rng):
    if not correlated:
        return variance * np.eye(p)
    A = rng.standard_normal((p, p))
    return variance * ((A @ A.T) / p + np.eye(p))


def _localization(n, localized, **radii):
    coords = ((np.arange(n) + 0.5) / n)[:, None]
    return Localization(horizontal=coords, horizontal_radius=LOCALIZATION_RADIUS, **radii) if localized else None


def make_problem(n, p, M, seed, *, localized=True, correlated_R=False, obs_variance=0.01) -> Problem:
    rng = np.random.default_rng(seed)
    fields = _random_fields(n, M + 1, rng)
    truth, Xf = fields[:, 0], 0.5 * fields[:, 1:]
    H = np.sort(rng.choice(n, size=p, replace=False))
    R = _covariance(p, obs_variance, correlated_R, rng)
    y = truth[H] + np.linalg.cholesky(R) @ rng.standard_normal(p)
    Y = y[:, None] + np.linalg.cholesky(R) @ rng.standard_normal((p, M))
    return Problem(Xf=Xf, H=H, Y=Y, R=R, localization=_localization(n, localized))


def make_window_problem(n, p, M, K, seed, *, localized=True, correlated_R=False, obs_variance=0.01) -> WindowProblem:
    """K window times, p observations per time; the error variance at time k is (k + 1) * obs_variance."""
    rng = np.random.default_rng(seed)
    w = _random_fields(n, K * (M + 1), rng).reshape(n, K, M + 1)      # column 0: truth, then the members
    x = [w[:, 0]]
    for k in range(1, K):
        x.append(TIME_CORRELATION * x[-1] + np.sqrt(1 - TIME_CORRELATION ** 2) * w[:, k])
    observations = []
    for k in range(K):
        H = np.sort(rng.choice(n, size=p, replace=False))
        R = _covariance(p, (k + 1) * obs_variance, correlated_R, rng)
        observations.append(Observations(x[k][H, 0] + np.linalg.cholesky(R) @ rng.standard_normal(p), H, R))
    return WindowProblem(ensembles=[0.5 * xk[:, 1:] for xk in x], observations=observations,
                         times=TIME_STEP * np.arange(K), localization=_localization(n, localized, time_radius=TIME_RADIUS))


def window_state_problem(wp: WindowProblem, seed) -> Problem:
    """The reduced window state of wp as one analysis problem, assembled as assimilate_window does."""
    ws = build_window_state(wp.ensembles, [o.indices for o in wp.observations])
    R = block_diag(*[o.covariance for o in wp.observations])
    y = np.concatenate([o.values for o in wp.observations])
    Y = perturbed_observations(y, R, ws.forecast.shape[1], np.random.default_rng(seed))
    return Problem(Xf=ws.forecast, H=ws.obs_rows, Y=Y, R=R, localization=wp.localization,
                   rows=(ws.component, wp.times[ws.time_index]))


def background_covariance(problem: Problem) -> np.ndarray:
    B = estimate_covariance(problem.Xf)
    return B if problem.localization is None else B * problem.localization.matrix(*problem.rows)
