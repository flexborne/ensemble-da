"""Reduced multi-time state of an analysis window (independent of any grid or model)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class WindowState:
    """An analysis window stacked into one state vector.

    Rows of `forecast` are (component, time) pairs: for each earlier time only the components it
    observes, then the full state at the target (last) time.
    """

    forecast: np.ndarray     # (N, M) stacked forecast ensemble
    component: np.ndarray    # (N,) state component of each row
    time_index: np.ndarray   # (N,) position in the window's time sequence of each row
    obs_rows: np.ndarray     # (p,) row observed by each observation, in window order
    target: slice            # rows holding the full state at the target time


def build_window_state(ensembles, observed) -> WindowState:
    """Stack an analysis window into a reduced state.

    ensembles : K forecast ensembles (n, M) at the window times, the target time last.
    observed  : K integer arrays — the state components observed at each time.

    Earlier times contribute one row per distinct observed component; the target time contributes
    the full state. Observations are ordered by time, as given. A one-time window is the filter.
    """
    if len(ensembles) != len(observed) or not ensembles:
        raise ValueError('need one observed-index array per ensemble, and at least one time')
    ensembles = [np.asarray(e, dtype=float) for e in ensembles]
    n, M = ensembles[-1].shape
    if any(e.shape != (n, M) for e in ensembles):
        raise ValueError('all window ensembles must have the same (n, M) shape')

    blocks, component, time_index, obs_rows = [], [], [], []
    offset = 0
    for k, (ens, idx) in enumerate(zip(ensembles, observed)):
        idx = np.asarray(idx, dtype=int).ravel()
        if idx.size and (idx.min() < 0 or idx.max() >= n):
            raise ValueError(f'observed index out of range 0..{n - 1} at window time {k}')
        rows = np.arange(n) if k == len(ensembles) - 1 else np.unique(idx)
        blocks.append(ens[rows])
        component.append(rows)
        time_index.append(np.full(rows.size, k))
        obs_rows.append(offset + np.searchsorted(rows, idx))
        offset += rows.size
    return WindowState(forecast=np.vstack(blocks), component=np.concatenate(component),
                       time_index=np.concatenate(time_index), obs_rows=np.concatenate(obs_rows),
                       target=slice(offset - n, offset))
