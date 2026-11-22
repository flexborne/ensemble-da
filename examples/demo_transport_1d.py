"""Twin experiment: forecast, analysis and error against a known truth, with no external data.

A 1-D advection-diffusion model (examples/transport1d.py, n = 200 cells) carries a positive tracer
field c = exp(0.5 g), where g is a Gaussian random field with correlation length 0.03. The truth
and each of the 20 ensemble members start from independent draws of that prior; the members run
with uncertain velocities (model error). Synthetic observations of the truth at every 4th cell
(p = 50) carry multiplicative log-normal errors, so the analysis works in log space with
R = sigma^2 I. With 20 members and 200 components, sample covariances carry spurious long-range
correlations, which covariance localization suppresses.

    python examples/demo_transport_1d.py [--seed 1] [--json results.json] [--plot demo.png]

Part 1: one analysis at t = 1.0 - forecast vs analysis error, with and without localization.
Part 2: sequential cycle over t = 0, 0.25, ..., 2.0 for noda (free run), filter and smoother.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transport1d import Transport1D  # noqa: E402

from ensemble_da import Localization, Observations, assimilate, observation_covariance, run_cycle  # noqa: E402

N_CELLS, N_MEMBERS = 200, 20
CORR_LENGTH, LOG_STD = 0.03, 0.5                  # prior tracer field: c = exp(LOG_STD * g)
TRUE_VELOCITY, VELOCITY_STD = 0.20, 0.02
OBS_EVERY, OBS_LOG_STD = 4, 0.10                  # observation network; log-space error (GSD = e^0.1)
LOCALIZATION_RADIUS, TIME_RADIUS = 0.08, 0.5
TIMES = np.round(np.arange(0.0, 2.0 + 1e-9, 0.25), 10)


def rmse(ensemble, truth) -> float:
    """Root-mean-square error of the ensemble mean against the truth."""
    return float(np.sqrt(np.mean((ensemble.mean(axis=1) - truth) ** 2)))


def setup(seed):
    rng = np.random.default_rng(seed)
    model = Transport1D(n=N_CELLS)
    truth0 = np.exp(LOG_STD * model.random_field(CORR_LENGTH, rng))
    ensemble0 = np.exp(LOG_STD * model.random_field(CORR_LENGTH, rng, size=N_MEMBERS))
    velocities = TRUE_VELOCITY + VELOCITY_STD * rng.standard_normal(N_MEMBERS)
    obs_idx = np.arange(OBS_EVERY // 2, N_CELLS, OBS_EVERY)
    R = observation_covariance(OBS_LOG_STD, n_obs=obs_idx.size)

    truths = {t: model.propagate(truth0, t, TRUE_VELOCITY) for t in TIMES}
    obs_rng = np.random.default_rng(seed + 1000)
    observations = {t: Observations(truths[t][obs_idx] * np.exp(OBS_LOG_STD * obs_rng.standard_normal(obs_idx.size)),
                                    obs_idx, R) for t in TIMES}

    def forecast(ensemble, t_from, t_to):
        return model.propagate(ensemble, t_to - t_from, velocities)

    localization = Localization(horizontal=model.x[:, None], horizontal_radius=LOCALIZATION_RADIUS,
                                time_radius=TIME_RADIUS)
    return ensemble0, truths, observations, forecast, localization


def run(seed=1) -> dict:
    ensemble0, truths, observations, forecast, localization = setup(seed)

    # Part 1: one analysis at t = 1.0
    t1 = 1.0
    xf = forecast(ensemble0, 0.0, t1)
    obs = observations[t1]
    single = {'forecast_rmse': rmse(xf, truths[t1])}
    for label, rho in (('analysis_rmse_localized', localization.matrix()), ('analysis_rmse_unlocalized', None)):
        res = assimilate(xf, obs.values, obs.indices, obs.covariance, localization=rho, transform='log', rng=seed)
        single[label] = rmse(res.analysis, truths[t1])

    # Part 2: sequential cycles
    cycles = {}
    for mode, window in (('noda', 0.0), ('filter', 0.0), ('smoother', 0.25)):
        out = run_cycle(forecast, ensemble0, TIMES, observations, mode=mode, window=window,
                        localization=localization, transform='log', rng=seed)
        cycles[mode] = {
            'forecast_rmse': {float(t): rmse(out.forecasts[t], truths[t]) for t in TIMES[1:]},
            'analysis_rmse': {float(t): rmse(out.analyses[t], truths[t]) for t in out.analyses},
        }
    return {'seed': seed, 'single_analysis_t1': single, 'cycle': cycles,
            'setup': {'n_cells': N_CELLS, 'n_members': N_MEMBERS, 'n_obs': int(observations[t1].values.size),
                      'obs_log_std': OBS_LOG_STD, 'localization_radius': LOCALIZATION_RADIUS,
                      'time_radius': TIME_RADIUS, 'times': [float(t) for t in TIMES]}}


def _report(res):
    s = res['single_analysis_t1']
    print('Part 1 - one analysis at t = 1.0 (RMSE of the ensemble mean vs truth)')
    print(f"  forecast                {s['forecast_rmse']:.4f}")
    print(f"  analysis, localized     {s['analysis_rmse_localized']:.4f}")
    print(f"  analysis, unlocalized   {s['analysis_rmse_unlocalized']:.4f}")
    print('\nPart 2 - sequential cycle (forecast RMSE at each time; analysis RMSE in brackets)')
    print('  time    noda              filter            smoother')
    for t in res['setup']['times'][1:]:
        cells = []
        for mode in ('noda', 'filter', 'smoother'):
            c = res['cycle'][mode]
            a = c['analysis_rmse'].get(t)
            cells.append(f"{c['forecast_rmse'][t]:.4f}" + (f" [{a:.4f}]" if a is not None else ''))
        print(f'  {t:4.2f}    ' + '  '.join(f'{x:<16}' for x in cells))
    mean = lambda d: float(np.mean(list(d.values())))  # noqa: E731
    print('  mean    ' + '  '.join(
        f"{mean(res['cycle'][m]['forecast_rmse']):.4f}"
        + (f" [{mean(res['cycle'][m]['analysis_rmse']):.4f}]" if res['cycle'][m]['analysis_rmse'] else '')
        for m in ('noda', 'filter', 'smoother')))


def _plot(res, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for mode in ('noda', 'filter', 'smoother'):
        f = res['cycle'][mode]['forecast_rmse']
        ax.plot(list(f), list(f.values()), marker='o', label=f'{mode} forecast')
    for mode in ('filter', 'smoother'):
        a = res['cycle'][mode]['analysis_rmse']
        ax.plot(list(a), list(a.values()), ls='--', marker='x', label=f'{mode} analysis')
    ax.set(xlabel='time', ylabel='RMSE of ensemble mean', yscale='log', title='1-D transport twin experiment')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--json', help='write the numbers to this JSON file')
    p.add_argument('--plot', help='save a figure (needs matplotlib)')
    args = p.parse_args(argv)
    res = run(args.seed)
    _report(res)
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
    if args.plot:
        _plot(res, args.plot)


if __name__ == '__main__':
    main()
