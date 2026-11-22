"""High-level analysis step on plain arrays: forecast ensemble + observations -> analysis ensemble."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import block_diag

from .analysis import analysis_update, apply_operator
from .covariance import Localization, estimate_covariance
from .observations import perturbed_observations
from .state import build_window_state
from .transform import from_log, to_log

TRANSFORMS = ('none', 'log')


@dataclass
class AssimilationResult:
    """Analysis ensemble (n, M) in the input space, plus diagnostics in the analysis space."""

    analysis: np.ndarray
    diagnostics: dict = field(default_factory=dict)


def _rms(v) -> float:
    return float(np.sqrt(np.mean(v ** 2))) if v.size else float('nan')


def _spread(Z) -> float:
    return float(np.sqrt(np.mean(Z.var(axis=1, ddof=1))))


def _diagnostics(Zf, Za, z, H) -> dict:
    return {
        'n_state': int(Zf.shape[0]), 'n_obs': int(z.size), 'n_members': int(Zf.shape[1]),
        'innovation_rms': _rms(z - apply_operator(H, Zf.mean(axis=1))),
        'residual_rms': _rms(z - apply_operator(H, Za.mean(axis=1))),
        'spread_forecast': _spread(Zf), 'spread_analysis': _spread(Za),
    }


def assimilate(forecast_ensemble, observations, observation_operator, observation_covariance, *,
               localization=None, transform='none', perturb=True, rng=None) -> AssimilationResult:
    """One ensemble Kalman analysis step. No files and no forecast model are involved.

    forecast_ensemble      : (n, M) forecast states, M >= 2.
    observations           : (p,) observed values y.
    observation_operator   : (p,) indices of the observed state components, or a (p, n) matrix H.
    observation_covariance : (p, p) error covariance R, in the analysis space
                             (for transform='log': the covariance of log y).
    localization           : optional Localization, or an (n, n) matrix rho; B = rho * cov(Xf).
    transform              : 'none', or 'log' to analyse positive quantities as log-normal. A matrix H
                             then acts on log x (it observes a weighted geometric mean).
    perturb, rng           : stochastic EnKF perturbations y + N(0, R); rng is a Generator or a seed.
    Diagnostics (analysis space): innovation and residual RMS of the ensemble mean, spreads, sizes.
    """
    if transform not in TRANSFORMS:
        raise ValueError(f'transform must be one of {TRANSFORMS} (got {transform!r})')
    Xf = np.asarray(forecast_ensemble, dtype=float)
    if Xf.ndim != 2 or Xf.shape[1] < 2:
        raise ValueError(f'forecast_ensemble must be (n, M) with M >= 2, got shape {Xf.shape}')
    y = np.asarray(observations, dtype=float).ravel()
    H = np.asarray(observation_operator)
    Zf = to_log(Xf, 'forecast') if transform == 'log' else Xf
    if y.size == 0:                                   # nothing to assimilate
        return AssimilationResult(analysis=Xf.copy(), diagnostics=_diagnostics(Zf, Zf, y, H))

    z = to_log(y, 'observations') if transform == 'log' else y
    B = estimate_covariance(Zf)
    if localization is not None:
        B = B * (localization.matrix() if isinstance(localization, Localization) else np.asarray(localization))
    R = np.asarray(observation_covariance, dtype=float)
    M = Xf.shape[1]
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
    times        : K window times; temporal localization is applied only when they are given.
    localization : a Localization with per-component coordinates (rows are (component, time) pairs).
    Returns the analysis (n, M) at the target time; a one-time window is an ordinary filter step.
    n_state and the spreads describe that target state; innovation and residual RMS cover every
    observation in the window.
    """
    observations = list(observations)
    ws = build_window_state(ensembles, [o.indices for o in observations])
    y = np.concatenate([o.values for o in observations])
    R = block_diag(*[o.covariance for o in observations])
    rho = None
    if localization is not None:
        rho = localization.matrix(ws.component, None if times is None else np.asarray(times, dtype=float)[ws.time_index])
    res = assimilate(ws.forecast, y, ws.obs_rows, R, localization=rho, transform=transform,
                     perturb=perturb, rng=rng)
    analysis = res.analysis[ws.target]
    Zf, Za = ws.forecast[ws.target], analysis
    if transform == 'log':
        Zf, Za = np.log(Zf), np.log(Za)
    diagnostics = {**res.diagnostics, 'n_state': int(Zf.shape[0]), 'spread_forecast': _spread(Zf),
                   'spread_analysis': _spread(Za), 'window_times': len(observations)}
    return AssimilationResult(analysis=analysis, diagnostics=diagnostics)
