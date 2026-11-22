"""Independent reference solver, for verification and benchmarking only (slow, state space).

It shares no code with `analysis_update`: the same minimization is solved in a different space
with different linear algebra (eigendecomposition + SVD least squares instead of a Cholesky
factorization of the p x p innovation covariance).
"""
from __future__ import annotations

import numpy as np


def reference_analysis(B, H, Y, R, Xf, rtol=1e-12) -> np.ndarray:
    """Kalman analysis as a weighted least-squares problem in state space.

    With B = S S^T (eigendecomposition; modes below rtol * max eigenvalue dropped) each member solves
        min_w ||w||^2 + ||R^(-1/2) (H xf + H S w - y)||^2,      xa = xf + S w,
    by numpy.linalg.lstsq on the stacked whitened system. This is the variational form of the
    update with the increment restricted to range(B), so singular (rank-deficient) B is allowed;
    by the push-through identity S w = B H^T (H B H^T + R)^-1 (y - H xf).

    Arguments as for analysis_update: B (n, n), H (p,) indices or (p, n) matrix, Y (p, M),
    R (p, p), Xf (n, M). Returns Xa (n, M).
    """
    B = np.asarray(B, dtype=float)
    R = np.asarray(R, dtype=float)
    Xf = np.asarray(Xf, dtype=float)
    Y = np.asarray(Y, dtype=float)
    Xf = Xf.reshape(-1, 1) if Xf.ndim == 1 else Xf
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    H = np.asarray(H)
    Hm = np.eye(B.shape[0])[H.astype(int)] if H.ndim == 1 else H.astype(float)

    lam, U = np.linalg.eigh(B)
    keep = lam > rtol * max(lam.max(), 0.0)
    S = U[:, keep] * np.sqrt(lam[keep])                     # (n, r), B ~= S S^T
    mu, V = np.linalg.eigh(R)
    R_inv_sqrt = (V / np.sqrt(mu)) @ V.T                    # symmetric R^(-1/2)

    r = S.shape[1]
    A = np.vstack([np.eye(r), R_inv_sqrt @ Hm @ S])          # (r + p, r)
    rhs = np.vstack([np.zeros((r, Xf.shape[1])), R_inv_sqrt @ (Y - Hm @ Xf)])
    W = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return Xf + S @ W
