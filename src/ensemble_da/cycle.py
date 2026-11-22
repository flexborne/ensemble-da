"""Model-independent sequential assimilation-forecast cycle: noda, filter and smoother.

The forecast model stays outside the package. The cycle receives a callable
    forecast(ensemble, t_from, t_to) -> ensemble
and decides when to analyse, which window of times each analysis uses and which state seeds each
forecast. A one-time window is the filter; a smoother window holds the neighbouring times within
+/- `window` of the analysis time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .assimilate import assimilate_window
from .observations import Observations
from .transform import to_log

MODES = ('noda', 'filter', 'smoother')


@dataclass(frozen=True)
class CycleStep:
    kind: str                     # 'analyze' | 'forecast'
    time: float                   # analyze: target time; forecast: destination time
    window: tuple = ()            # analyze: window times, the target last (length 1 = filter)
    source: float | None = None   # forecast: start time
    restart: str | None = None    # forecast: seeded from the 'analysis' or the 'forecast' at source


@dataclass
class CycleResult:
    steps: list = field(default_factory=list)       # executed step records, in order
    analyses: dict = field(default_factory=dict)    # time -> analysis ensemble (n, M)
    forecasts: dict = field(default_factory=dict)   # time -> latest forecast (times[0]: initial ensemble)


def derive_windows(times, window):
    """0-based inclusive (first, last) indices of the times lying within +/- window of each time."""
    t = np.asarray(times, dtype=float)
    tol = 1e-9 + 16 * np.spacing(np.abs(t).max())     # holds for large time values too
    first = [int(np.flatnonzero(t >= ti - window - tol)[0]) for ti in t]
    last = [int(np.flatnonzero(t <= ti + window + tol)[-1]) for ti in t]
    return first, last


def plan_cycle(mode, times, window=0.0) -> list:
    """Deterministic analyze/forecast plan, built in one pass over the times.

    For every time but the last: analyse it (filter: that time alone; smoother: its +/- window),
    then forecast to the next time seeded from that analysis (noda: from the previous forecast).
    A smoother keeps forecasting to the end of the next window so its future states exist, and
    free-runs from the initial ensemble to the end of the first window before the first analysis.
    """
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES} (got {mode!r})')
    times = [float(t) for t in times]
    steps = []
    if mode == 'smoother':
        first, last = derive_windows(times, window)
        steps += [CycleStep('forecast', times[f], source=times[f - 1], restart='forecast')
                  for f in range(1, last[0] + 1)]
    for t in range(len(times) - 1):
        if mode == 'filter':
            steps.append(CycleStep('analyze', times[t], window=(times[t],)))
        elif mode == 'smoother':
            neighbours = times[first[t]:t] + times[t + 1:last[t] + 1]
            steps.append(CycleStep('analyze', times[t], window=tuple(neighbours + [times[t]])))
        horizon_end = last[t + 1] + 1 if mode == 'smoother' else t + 2    # exclusive time index
        for f in range(t + 1, horizon_end):
            if f == t + 1:
                restart = 'forecast' if mode == 'noda' else 'analysis'
                steps.append(CycleStep('forecast', times[f], source=times[t], restart=restart))
            else:
                steps.append(CycleStep('forecast', times[f], source=times[f - 1], restart='forecast'))
    return steps


def _innovation_rms(ensemble, obs, transform):
    mean = (to_log(ensemble, 'forecast') if transform == 'log' else ensemble).mean(axis=1)
    y = to_log(obs.values, 'observations') if transform == 'log' else obs.values
    return float(np.sqrt(np.mean((y - mean[obs.indices]) ** 2)))


def run_cycle(forecast, initial_ensemble, times, observations, *, mode='filter', window=0.0,
              localization=None, transform='none', perturb=True, rng=None) -> CycleResult:
    """Run a sequential cycle with an external forecast model.

    forecast         : callable(ensemble (n, M), t_from, t_to) -> ensemble (n, M).
    initial_ensemble : (n, M) ensemble at times[0].
    times            : at least two strictly increasing times.
    observations     : {time: Observations}; a time without an entry has no observations.
    mode             : 'noda' (forecast only), 'filter', or 'smoother' (+/- window).
    localization     : optional Localization (per-component coordinates; time_radius for windows).
    rng              : Generator or seed; one generator serves the whole cycle.
    Every time but the last is analysed (filter/smoother); the last is forecast-only.
    """
    times = [float(t) for t in times]
    if len(times) < 2 or not np.all(np.isfinite(times)) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('times must be at least two finite, strictly increasing values')
    unknown = set(observations) - set(times)
    if unknown:
        raise ValueError(f'observations given at times outside the cycle: {sorted(unknown)}')
    plan = plan_cycle(mode, times, window)
    rng = np.random.default_rng(rng)
    states = {times[0]: np.array(initial_ensemble, dtype=float)}   # stored states are always copies
    analyses, records = {}, []
    for i, step in enumerate(plan):
        if step.kind == 'analyze':
            obs = [observations.get(t, Observations.empty()) for t in step.window]
            res = assimilate_window([states[t] for t in step.window], obs, step.window,
                                    localization=localization, transform=transform, perturb=perturb, rng=rng)
            analyses[step.time] = res.analysis
            records.append({'id': i, 'step': 'analyze', 'time': step.time, 'window': list(step.window),
                            'diagnostics': res.diagnostics})
        else:
            start = analyses[step.source] if step.restart == 'analysis' else states[step.source]
            out = np.array(forecast(start.copy(), step.source, step.time), dtype=float)
            if out.shape != start.shape:
                raise ValueError(f'forecast returned shape {out.shape}, expected {start.shape}')
            states[step.time] = out
            rec = {'id': i, 'step': 'forecast', 'time': step.time, 'source': step.source, 'restart': step.restart}
            if step.time in observations and observations[step.time].values.size:
                rec['innovation_rms'] = _innovation_rms(out, observations[step.time], transform)
            records.append(rec)
    return CycleResult(steps=records, analyses=analyses, forecasts=states)
