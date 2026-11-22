"""The analysis update: an ensemble Kalman step solved in observation space."""
from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def apply_operator(H, X):
    """H X for an observation operator given as observed state indices (p,) or a (p, n) matrix."""
    H = np.asarray(H)
    return X[H] if H.ndim == 1 else H @ X


def analysis_update(B, H, Y, R, Xf):
    """Ensemble Kalman analysis computed in observation space.

    For every ensemble member m this minimizes
        J(x) = (x - xf_m)^T B^-1 (x - xf_m) + (H x - y_m)^T R^-1 (H x - y_m),
    whose minimizer is
        Xa = Xf + B H^T (H B H^T + R)^-1 (Y - H Xf),
    computed with one Cholesky factorization of the p x p innovation covariance. B may be
    singular (an unlocalized ensemble covariance); the increment then lies in the range of B.
    The relative accuracy of the increment is about machine epsilon times cond(H B H^T + R).

    B  : (n, n) background covariance, symmetric positive semi-definite.
    H  : (p,) integer indices of the observed state components, or a (p, n) matrix.
    Y  : (p, M) observations per member (perturbed if desired).
    R  : (p, p) observation-error covariance, symmetric positive definite.
    Xf : (n, M) forecast ensemble.  Returns Xa, (n, M).
    """
    B = np.asarray(B, dtype=float)
    R = np.asarray(R, dtype=float)
    Xf = np.asarray(Xf, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Xf = Xf.reshape(-1, 1) if Xf.ndim == 1 else Xf
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    n, p = Xf.shape[0], Y.shape[0]
    if R.shape != (p, p):                            # a vector R would broadcast silently
        raise ValueError(f'observation covariance must be ({p}, {p}), got {R.shape}')
    H = np.asarray(H)
    if H.ndim == 1:                                  # selection: observation k sees component H[k]
        if H.size and (not np.issubdtype(H.dtype, np.integer) or H.min() < 0 or H.max() >= n):
            raise ValueError(f'observation indices must be integers in 0..{n - 1}')
        BHt = B[:, H]
        S = B[np.ix_(H, H)] + R
    else:                                            # general linear observation operator
        BHt = B @ H.T
        S = H @ BHt + R
    Xa = Xf + BHt @ cho_solve(cho_factor(S, lower=True), Y - apply_operator(H, Xf))
    if not np.all(np.isfinite(Xa)):
        raise FloatingPointError('analysis_update produced non-finite values')
    return Xa
