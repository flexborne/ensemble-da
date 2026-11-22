"""Ensemble covariance estimation and distance-based covariance localization."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist

EARTH_RADIUS_M = 6_371_000.0
METRICS = ('euclidean', 'great_circle')


def estimate_covariance(X) -> np.ndarray:
    """Sample covariance of an ensemble X (n, M): rows are state components, 1/(M-1) normalization."""
    X = np.asarray(X, dtype=float)
    A = X - X.mean(axis=1, keepdims=True)
    return A @ A.T / (X.shape[1] - 1)


def great_circle_distance(a, b) -> np.ndarray:
    """Pairwise great-circle distances in metres between (k, 2) and (l, 2) [lat, lon] arrays (degrees)."""
    a, b = np.radians(np.asarray(a, dtype=float)), np.radians(np.asarray(b, dtype=float))
    dlat = b[None, :, 0] - a[:, None, 0]
    dlon = b[None, :, 1] - a[:, None, 1]
    h = np.sin(dlat / 2) ** 2 + np.cos(a[:, None, 0]) * np.cos(b[None, :, 0]) * np.sin(dlon / 2) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


@dataclass(frozen=True, eq=False)
class Localization:
    """Distance-based localization factors for an ensemble covariance (Schur product B = rho * C).

        rho_ij = exp(-d_h(i, j) / L_h) * exp(-d_v(i, j) / L_v) * exp(-|t_i - t_j| / L_t)

    Coordinates are given once per state component (length n); a factor is applied only when both
    its coordinates and its radius are set. Components flagged in `decoupled` keep their variance
    but lose every covariance with other components (including themselves at other times).
    """

    horizontal: np.ndarray | None = None      # (n, d) positions; [lat, lon] degrees for great_circle
    vertical: np.ndarray | None = None        # (n,) vertical coordinate, e.g. depth
    horizontal_radius: float | None = None    # same unit as the horizontal distance (m for great_circle)
    vertical_radius: float | None = None
    time_radius: float | None = None          # used by multi-time (smoother) windows
    metric: str = 'euclidean'                 # 'euclidean' or 'great_circle'
    decoupled: np.ndarray | None = None       # (n,) bool

    def __post_init__(self):
        if self.metric not in METRICS:
            raise ValueError(f'metric must be one of {METRICS} (got {self.metric!r})')
        for name in ('horizontal_radius', 'vertical_radius', 'time_radius'):
            r = getattr(self, name)
            if r is not None and not (np.isfinite(r) and r > 0):
                raise ValueError(f'{name} must be a finite positive value (got {r})')

    def _size(self) -> int | None:
        for arr in (self.horizontal, self.vertical, self.decoupled):
            if arr is not None:
                return len(arr)
        return None

    def matrix(self, components=None, times=None) -> np.ndarray:
        """Localization matrix for the given state components (default: all of them), each
        optionally tagged with a time; repeated components at different times are allowed."""
        if components is None:
            n = self._size()
            if n is None:
                raise ValueError('pass components: no per-component coordinates to infer n from')
            components = np.arange(n)
        comp = np.asarray(components, dtype=int)
        k = comp.size
        exponent = np.zeros((k, k))
        if self.horizontal is not None and self.horizontal_radius:
            pos = np.asarray(self.horizontal, dtype=float)[comp].reshape(k, -1)
            dist = great_circle_distance(pos, pos) if self.metric == 'great_circle' else cdist(pos, pos)
            exponent += dist / self.horizontal_radius
        if self.vertical is not None and self.vertical_radius:
            z = np.asarray(self.vertical, dtype=float)[comp]
            exponent += np.abs(z[:, None] - z[None, :]) / self.vertical_radius
        if times is not None and self.time_radius:
            t = np.asarray(times, dtype=float)
            exponent += np.abs(t[:, None] - t[None, :]) / self.time_radius
        rho = np.exp(-exponent)
        if self.decoupled is not None:
            idx = np.flatnonzero(np.asarray(self.decoupled, dtype=bool)[comp])
            rho[idx, :] = 0.0
            rho[:, idx] = 0.0
            rho[idx, idx] = 1.0
        return rho
