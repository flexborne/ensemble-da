"""High-level analysis step on plain arrays: forecast ensemble + observations -> analysis ensemble."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import block_diag

from .analysis import analysis_update, apply_operator
from .covariance import Localization, estimate_covariance
from .observations import Observations, perturbed_observations
from .state import build_window_state
from .transform import from_log, to_log

TRANSFORMS = ('none', 'log')


@dataclass
class AssimilationResult:
    """Analysis ensemble (n, M) in the input space, plus diagnostics in the analysis space."""

    analysis: np.ndarray
    diagnostics: dict = field(default_factory=dict)


def _diagnostics(Zf, Za, y, H) -> dict:
    innovation = y - apply_operator(H, Zf.mean(axis=1))
    residual = y - apply_operator(H, Za.mean(axis=1))
    return {
        'n_state': int(Zf.shape[0]), 'n_obs': int(y.size), 'n_members': int(Zf.shape[1]),
        'innovation_rms': float(np.sqrt(np.mean(innovation ** 2))),
        'residual_rms': float(np.sqrt(np.mean(residual ** 2))),
        'spread_forecast': float(np.sqrt(np.mean(Zf.var(axis=1, ddof=1)))),
        'spread_analysis': float(np.sqrt(np.mean(Za.var(axis=1, ddof=1)))),
    }


def assimilate(forecast_ensemble, observations, observation_operator, observation_covariance, *,
               localization=None, transform='none', perturb=True, rng=None) -> AssimilationResult:
    """One ensemble Kalman analysis step. No files and no forecast model are involved.

    forecast_ensemble      : (n, M) forecast states, M >= 2.
    observations           : (p,) observed values y.
    observation_operator   : (p,) indices of the observed state components, or a (p, n) matrix H.
    observation_covariance : (p, p) error covariance R, in the analysis space
                             (for transform='log': the covariance of log y).
    localization           : optional (n, n) factors rho; B = rho * cov(Xf) (see Localization.matrix).
    transform              : 'none', or 'log' to analyse positive quantities as log-normal.
    perturb, rng           : stochastic EnKF perturbations y + N(0, R); rng is a Generator or a seed.
    """
    if transform not in TRANSFORMS:
        raise ValueError(f'transform must be one of {TRANSFORMS} (got {transform!r})')
    Xf = np.asarray(forecast_ensemble, dtype=float)
    if Xf.ndim != 2 or Xf.shape[1] < 2:
        raise ValueError(f'forecast_ensemble must be (n, M) with M >= 2, got shape {Xf.shape}')
    n, M = Xf.shape
    y = np.asarray(observations, dtype=float).ravel()
    p = y.size
    H = np.asarray(observation_operator)
    if H.ndim == 1:
        if H.size != p or (p and (not np.issubdtype(H.dtype, np.integer) or H.min() < 0 or H.max() >= n)):
            raise ValueError(f'observation indices must be {p} integers in 0..{n - 1}')
    elif H.shape != (p, n):
        raise ValueError(f'observation operator must be ({p}, {n}), got {H.shape}')
    R = np.asarray(observation_covariance, dtype=float)
    if R.shape != (p, p):
        raise ValueError(f'observation covariance must be ({p}, {p}), got {R.shape}')
    if p == 0:                                        # nothing to assimilate
        return AssimilationResult(analysis=Xf.copy(), diagnostics={'n_state': n, 'n_obs': 0, 'n_members': M})

    Zf, z = (to_log(Xf, 'forecast'), to_log(y, 'observations')) if transform == 'log' else (Xf, y)
    B = estimate_covariance(Zf)
    if localization is not None:
        rho = np.asarray(localization, dtype=float)
        if rho.shape != (n, n):
            raise ValueError(f'localization must be ({n}, {n}), got {rho.shape}')
        B = B * rho
    Y = perturbed_observations(z, R, M, np.random.default_rng(rng)) if perturb else np.repeat(z[:, None], M, axis=1)
    Za = analysis_update(B, H, Y, R, Zf)
    Xa = from_log(Za, 'analysis') if transform == 'log' else Za
    return AssimilationResult(analysis=Xa, diagnostics=_diagnostics(Zf, Za, z, H))


def assimilate_window(ensembles, observations, times=None, *, localization=None,
                      transform='none', perturb=True, rng=None) -> AssimilationResult:
    """Smoother analysis of the last time of a window, using the observations at every window time.

    ensembles    : K forecast ensembles (n, M) at the window times, the target time last.
    observations : K Observations records (empty ones allowed); errors at different times are
                   independent, so R is block-diagonal.
    times        : K window times, needed only for temporal localization.
    localization : a Localization with per-component coordinates (not a precomputed matrix), so
                   the window's (component, time) rows can be localized.
    Returns the analysis (n, M) at the target time; a one-time window is an ordinary filter step.
    """
    observations = list(observations)
    if not all(isinstance(o, Observations) for o in observations):
        raise TypeError('observations must be Observations records, one per window time')
    ws = build_window_state(ensembles, [o.indices for o in observations])
    y = np.concatenate([o.values for o in observations])
    R = block_diag(*[o.covariance for o in observations])
    rho = None
    if localization is not None:
        if not isinstance(localization, Localization):
            raise TypeError('window localization must be a Localization (per-component coordinates)')
        row_times = None if times is None else np.asarray(times, dtype=float)[ws.time_index]
        rho = localization.matrix(ws.component, row_times)
    res = assimilate(ws.forecast, y, ws.obs_rows, R, localization=rho, transform=transform,
                     perturb=perturb, rng=rng)
    return AssimilationResult(analysis=res.analysis[ws.target],
                              diagnostics={**res.diagnostics, 'window_times': len(observations)})
