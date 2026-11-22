"""Twin experiment: forecast, analysis and error against a known truth, with no external data.

A 1-D periodic advection-diffusion model (examples/transport1d.py, n = 200 cells) carries a
positive tracer field c = exp(0.5 g), where g is a Gaussian random field with correlation length
0.03. The truth and each of the 20 ensemble members start from independent draws of that prior
and advect with velocities drawn from the same prior N(0.2, 0.02) (model error). Synthetic
observations of the truth at every 4th cell (p = 50) carry multiplicative log-normal errors, so the
analysis works in log space with R = sigma^2 I. Truth, ensemble, observations and assimilation use
independent random streams. With 20 members and 200 components, sample covariances carry spurious
long-range correlations, which covariance localization suppresses.

    python examples/demo_transport_1d.py [--seed 1] [--seeds 20] [--json out.json] [--plot out.png]

Part 1: one analysis at t = 1.0 - forecast vs analysis, with and without localization.
Part 2: sequential cycle over t = 0, 0.25, ..., 2.0 for noda (free run), filter and smoother.
--seeds N repeats the experiment for seeds 1..N and summarizes the error ratios.
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
VELOCITY_MEAN, VELOCITY_STD = 0.20, 0.02          # prior of the advection velocity (truth and members)
OBS_EVERY, OBS_LOG_STD = 4, 0.10                  # observation network; log-space error (GSD = e^0.1)
LOCALIZATION_RADIUS, TIME_RADIUS = 0.08, 0.5
TIMES = np.linspace(0.0, 2.0, 9)                  # every 0.25
T_SINGLE = 1.0


def rmse(ensemble, truth) -> float:
    """Root-mean-square error of the ensemble mean against the truth."""
    return float(np.sqrt(np.mean((ensemble.mean(axis=1) - truth) ** 2)))


def setup(seed):
    truth_seq, ens_seq, obs_seq, da_seq = np.random.SeedSequence(seed).spawn(4)
    truth_rng, ens_rng, obs_rng = (np.random.default_rng(s) for s in (truth_seq, ens_seq, obs_seq))
    model = Transport1D(n=N_CELLS)
    truth0 = np.exp(LOG_STD * model.random_field(CORR_LENGTH, truth_rng))
    true_velocity = VELOCITY_MEAN + VELOCITY_STD * truth_rng.standard_normal()
    ensemble0 = np.exp(LOG_STD * model.random_field(CORR_LENGTH, ens_rng, size=N_MEMBERS))
    velocities = VELOCITY_MEAN + VELOCITY_STD * ens_rng.standard_normal(N_MEMBERS)

    obs_idx = np.arange(OBS_EVERY // 2, N_CELLS, OBS_EVERY)
    R = observation_covariance(OBS_LOG_STD, n_obs=obs_idx.size)
    truths = {t: model.propagate(truth0, t, true_velocity) for t in TIMES}
    observations = {t: Observations(truths[t][obs_idx] * np.exp(OBS_LOG_STD * obs_rng.standard_normal(obs_idx.size)),
                                    obs_idx, R) for t in TIMES}

    def forecast(ensemble, t_from, t_to):
        return model.propagate(ensemble, t_to - t_from, velocities)

    angle = 2 * np.pi * model.x / model.length                     # periodic domain: chord distance on a circle
    circle = model.length / (2 * np.pi) * np.column_stack([np.cos(angle), np.sin(angle)])
    localization = Localization(horizontal=circle, horizontal_radius=LOCALIZATION_RADIUS, time_radius=TIME_RADIUS)
    return model, ensemble0, truths, observations, forecast, localization, da_seq


def run(seed=1) -> dict:
    model, ensemble0, truths, observations, forecast, localization, da_seq = setup(seed)

    # Part 1: one analysis at T_SINGLE (every run restarts the same assimilation stream da_seq)
    xf = forecast(ensemble0, 0.0, T_SINGLE)
    obs = observations[T_SINGLE]
    localized = assimilate(xf, obs.values, obs.indices, obs.covariance, localization=localization,
                           transform='log', rng=da_seq).analysis
    unlocalized = assimilate(xf, obs.values, obs.indices, obs.covariance, transform='log', rng=da_seq).analysis
    single = {'forecast_rmse': rmse(xf, truths[T_SINGLE]),
              'analysis_rmse_localized': rmse(localized, truths[T_SINGLE]),
              'analysis_rmse_unlocalized': rmse(unlocalized, truths[T_SINGLE])}
    fields = {'x': model.x.tolist(), 'truth': truths[T_SINGLE].tolist(), 'forecast_mean': xf.mean(axis=1).tolist(),
              'analysis_mean': localized.mean(axis=1).tolist(),
              'obs_x': model.x[obs.indices].tolist(), 'obs': obs.values.tolist()}

    # Part 2: sequential cycles
    cycles = {}
    for mode, window in (('noda', 0.0), ('filter', 0.0), ('smoother', 0.25)):
        out = run_cycle(forecast, ensemble0, TIMES, observations, mode=mode, window=window,
                        localization=localization, transform='log', rng=da_seq)
        cycles[mode] = {
            'forecast_rmse': {float(t): rmse(out.forecasts[t], truths[t]) for t in TIMES[1:]},
            'analysis_rmse': {float(t): rmse(out.analyses[t], truths[t]) for t in out.analyses},
        }
    return {'seed': seed, 'single_analysis': single, 'fields': fields, 'cycle': cycles,
            'setup': {'n_cells': N_CELLS, 'n_members': N_MEMBERS, 'n_obs': int(obs.values.size),
                      'obs_log_std': OBS_LOG_STD, 'localization_radius': LOCALIZATION_RADIUS,
                      'time_radius': TIME_RADIUS, 't_single': T_SINGLE, 'times': [float(t) for t in TIMES]}}


def matched_mean(rmse_by_time) -> float:
    """Mean RMSE over the times that have both a forecast and an analysis (all but the first and last)."""
    return float(np.mean([rmse_by_time[float(t)] for t in TIMES[1:-1]]))


def ratios(res) -> dict:
    """Error ratios that summarize one experiment (< 1 means the first method is better)."""
    s, c = res['single_analysis'], res['cycle']
    f = {mode: matched_mean(c[mode]['forecast_rmse']) for mode in ('noda', 'filter', 'smoother')}
    a = {mode: matched_mean(c[mode]['analysis_rmse']) for mode in ('filter', 'smoother')}
    return {
        'analysis / forecast (one analysis)': s['analysis_rmse_localized'] / s['forecast_rmse'],
        'localized / unlocalized analysis': s['analysis_rmse_localized'] / s['analysis_rmse_unlocalized'],
        'filter / free-run forecast': f['filter'] / f['noda'],
        'smoother / free-run forecast': f['smoother'] / f['noda'],
        'filter analysis / its forecast': a['filter'] / f['filter'],
        'smoother analysis / its forecast': a['smoother'] / f['smoother'],
        'smoother / filter forecast': f['smoother'] / f['filter'],
        'smoother / filter analysis': a['smoother'] / a['filter'],
    }


def summarize(seeds) -> dict:
    per_seed = [ratios(run(s)) for s in seeds]
    return {key: {'median': float(np.median([r[key] for r in per_seed])),
                  'min': float(min(r[key] for r in per_seed)), 'max': float(max(r[key] for r in per_seed)),
                  'below_1': int(sum(r[key] < 1 for r in per_seed)), 'seeds': len(per_seed)}
            for key in per_seed[0]}


def _report(res):
    s = res['single_analysis']
    print(f"Part 1 - one analysis at t = {T_SINGLE} (RMSE of the ensemble mean vs truth)")
    print(f"  forecast                {s['forecast_rmse']:.4f}")
    print(f"  analysis, localized     {s['analysis_rmse_localized']:.4f}")
    print(f"  analysis, unlocalized   {s['analysis_rmse_unlocalized']:.4f}")
    print('\nPart 2 - sequential cycle: forecast RMSE at each time, analysis RMSE in brackets')
    print('  time    noda      filter              smoother')
    c = res['cycle']
    for t in TIMES.tolist():
        row = [f"{c['noda']['forecast_rmse'][t]:.4f}" if t in c['noda']['forecast_rmse'] else '  -   ']
        for mode in ('filter', 'smoother'):
            f, a = c[mode]['forecast_rmse'].get(t), c[mode]['analysis_rmse'].get(t)
            row.append((f'{f:.4f}' if f is not None else '  -   ') + (f' [{a:.4f}]' if a is not None else ''))
        print(f'  {t:4.2f}    {row[0]:<8}  {row[1]:<18}  {row[2]}')
    print('  mean over t = 0.25-1.75: ' + '  '.join(
        f"{m} {matched_mean(c[m]['forecast_rmse']):.4f}"
        + (f" [{matched_mean(c[m]['analysis_rmse']):.4f}]" if m != 'noda' else '') for m in ('noda', 'filter', 'smoother')))


def _report_summary(summary):
    n = next(iter(summary.values()))['seeds']
    print(f'\nSummary over {n} seeds (ratio < 1: first method better)')
    for key, v in summary.items():
        print(f"  {key:<36} median {v['median']:.3f}  range {v['min']:.3f}-{v['max']:.3f}  below 1 in {v['below_1']}/{n}")


def _plot(res, path):
    import matplotlib.pyplot as plt
    f, c = res['fields'], res['cycle']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    ax1.plot(f['x'], f['truth'], 'k', lw=1.5, label='truth')
    ax1.plot(f['x'], f['forecast_mean'], color='tab:blue', label='forecast mean')
    ax1.plot(f['x'], f['analysis_mean'], color='tab:red', ls='--', label='analysis mean')
    ax1.plot(f['obs_x'], f['obs'], 'o', ms=3, color='tab:green', label='observations')
    ax1.set(xlabel='x', ylabel='tracer', title=f'fields at t = {T_SINGLE}')
    ax1.legend(fontsize=8)
    for mode in ('noda', 'filter', 'smoother'):
        ax2.plot(list(c[mode]['forecast_rmse']), list(c[mode]['forecast_rmse'].values()), marker='o', label=f'{mode} forecast')
    for mode in ('filter', 'smoother'):
        ax2.plot(list(c[mode]['analysis_rmse']), list(c[mode]['analysis_rmse'].values()), ls='--', marker='x',
                 label=f'{mode} analysis')
    ax2.set(xlabel='time', ylabel='RMSE of ensemble mean', yscale='log', title='sequential cycle')
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--seeds', type=int, default=0, help='also summarize seeds 1..N')
    p.add_argument('--json', help='write the numbers to this JSON file')
    p.add_argument('--plot', help='save a figure (needs matplotlib)')
    args = p.parse_args(argv)
    res = run(args.seed)
    _report(res)
    if args.seeds:
        res['summary'] = summarize(range(1, args.seeds + 1))
        _report_summary(res['summary'])
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))
    if args.plot:
        _plot(res, args.plot)


if __name__ == '__main__':
    main()
