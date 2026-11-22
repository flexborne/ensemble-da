"""Optional log transform for positive, log-normally distributed quantities."""
from __future__ import annotations

import numpy as np


def to_log(X, what='values') -> np.ndarray:
    """Natural log, rejecting non-positive or non-finite input (it has no log-normal meaning)."""
    X = np.asarray(X, dtype=float)
    bad = ~(np.isfinite(X) & (X > 0))
    if np.any(bad):
        where = tuple(int(i) for i in np.argwhere(bad)[0])
        raise ValueError(f'log transform needs finite positive {what}; got {X[where]} at {where}')
    return np.log(X)


def from_log(Z, what='values') -> np.ndarray:
    """exp back to physical space, rejecting overflow to infinity and underflow to zero."""
    X = np.exp(np.asarray(Z, dtype=float))
    bad = ~(np.isfinite(X) & (X > 0))
    if np.any(bad):
        where = tuple(int(i) for i in np.argwhere(bad)[0])
        raise FloatingPointError(f'{what} is not representable as a positive number at {where}: exp gives {X[where]}')
    return X
