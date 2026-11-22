"""Timing of the analysis step: observation-space vs state-space solution of the same problem.

Three solvers on identical, seeded inputs:
  obs_cholesky    ensemble_da.analysis_update      Cholesky of the p x p matrix H B H^T + R
  state_cholesky  baselines.state_space_cholesky   state-space (regression) form, n x n Cholesky
  state_lstsq     ensemble_da.reference_analysis   state-space weighted least squares (eigh + SVD)
The state-space solvers are re-implementations of those formulations, not measurements of any
other program. state_cholesky is the fair baseline; state_lstsq is the verification reference.

Configurations: filter steps (one time, state size n) and smoother windows (K times; the reduced
window state of N = n + earlier observed components, temporal localization). Stages timed
separately: covariance estimation, localization (factors + Schur product), solve (per solver), full
analysis (covariance + localization + solve, per solver).

    python benchmarks/benchmark_analysis.py [--threads 1] [--repeats 30] [--warmup 3] [--quick] [--out PATH]

BLAS threads are fixed through environment variables set before NumPy is imported. Samples of
all stages are interleaved; each sample averages enough calls to last >= 5 ms. Writes <out>.json
(environment, protocol, all statistics) and <out>.csv (one row per configuration/stage/method).
"""
from __future__ import annotations

import argparse
import os
import sys

THREAD_VARS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
               'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def _parse(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--threads', type=int, default=1, help='BLAS threads (default 1)')
    p.add_argument('--repeats', type=int, default=30, help='timed samples per stage (default 30)')
    p.add_argument('--warmup', type=int, default=3, help='untimed warm-up calls per stage (default 3)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--quick', action='store_true', help='one small configuration, for smoke tests')
    p.add_argument('--out', default=None, help='output path without extension')
    return p.parse_args(argv)


if __name__ == '__main__':                     # must happen before NumPy loads its BLAS
    ARGS = _parse(sys.argv[1:])
    for _var in THREAD_VARS:
        os.environ[_var] = str(ARGS.threads)

import csv  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import state_space_cholesky  # noqa: E402
from environment import environment  # noqa: E402
from problems import (CORR_LENGTH, LOCALIZATION_RADIUS, TIME_RADIUS, TIME_STEP, background_covariance,  # noqa: E402
                      make_problem, make_window_problem, window_state_problem)

from ensemble_da import analysis_update, estimate_covariance, reference_analysis  # noqa: E402

FILTER_CONFIGS = [  # (n, p, M)
    (80, 8, 24), (160, 16, 24), (160, 32, 24), (320, 32, 24), (640, 64, 24),
    (320, 32, 12), (320, 32, 48), (320, 32, 96),      # ensemble-size sweep
]
SMOOTHER_CONFIGS = [  # (n, p per window time, M, K)
    (320, 32, 24, 2), (320, 32, 24, 3), (640, 64, 24, 3),
]
SOLVERS = {'obs_cholesky': analysis_update, 'state_cholesky': state_space_cholesky, 'state_lstsq': reference_analysis}
SAMPLE_SECONDS = 0.005


def _calls_per_sample(fn):
    k = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(k):
            fn()
        if time.perf_counter() - t0 >= SAMPLE_SECONDS or k >= 4096:
            return k
        k *= 2


def _measure(fns, repeats, warmup):
    """Interleaved timing: every repeat takes one sample of each stage in turn."""
    for fn in fns.values():
        for _ in range(warmup):
            fn()
    calls = {name: _calls_per_sample(fn) for name, fn in fns.items()}
    wall = {name: [] for name in fns}
    cpu = {name: [] for name in fns}
    gc.collect()
    gc.disable()
    try:
        for _ in range(repeats):
            for name, fn in fns.items():
                c0, t0 = time.process_time(), time.perf_counter()
                for _ in range(calls[name]):
                    fn()
                dt = time.perf_counter() - t0
                wall[name].append(dt / calls[name])
                cpu[name].append((time.process_time() - c0) / dt)
    finally:
        gc.enable()
    stats = {}
    for name, s in wall.items():
        s = np.asarray(s)
        stats[name] = {'median_s': float(np.median(s)), 'p25_s': float(np.percentile(s, 25)),
                       'p75_s': float(np.percentile(s, 75)), 'min_s': float(s.min()), 'max_s': float(s.max()),
                       'repeats': repeats, 'calls_per_sample': calls[name],
                       'cpu_wall_ratio': float(np.median(cpu[name]))}
    return stats


def run_config(prob, args):
    args_of = lambda B: (B, prob.H, prob.Y, prob.R, prob.Xf)  # noqa: E731
    C = estimate_covariance(prob.Xf)
    B = background_covariance(prob)
    analyses = {name: solve(*args_of(B)) for name, solve in SOLVERS.items()}
    ref = analyses['obs_cholesky']

    fns = {'covariance/shared': lambda: estimate_covariance(prob.Xf),
           'localization/shared': lambda: C * prob.localization.matrix(*prob.rows)}
    for name, solve in SOLVERS.items():
        fns[f'solve/{name}'] = lambda solve=solve: solve(*args_of(B))
        fns[f'full/{name}'] = lambda solve=solve: solve(*args_of(background_covariance(prob)))
    timings = _measure(fns, args.repeats, args.warmup)
    speedup = {stage: {name: timings[f'{stage}/{name}']['median_s'] / timings[f'{stage}/obs_cholesky']['median_s']
                       for name in ('state_cholesky', 'state_lstsq')} for stage in ('solve', 'full')}
    max_abs_diff = {name: float(np.abs(a - ref).max()) for name, a in analyses.items() if name != 'obs_cholesky'}
    return {'timings': timings, 'speedup_median': speedup, 'max_abs_diff': max_abs_diff}


def configs(quick, seed):
    """(labels, problem) per timed configuration: filter steps, then smoother windows."""
    for n, p, M in ([(80, 8, 24)] if quick else FILTER_CONFIGS):
        yield {'mode': 'filter', 'n': n, 'K': 1, 'p': p, 'M': M}, make_problem(n, p, M, seed)
    for n, p, M, K in ([(80, 8, 24, 2)] if quick else SMOOTHER_CONFIGS):
        yield ({'mode': 'smoother', 'n': n, 'K': K, 'p': p * K, 'M': M},
               window_state_problem(make_window_problem(n, p, M, K, seed), seed))


def main():
    args = ARGS
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / 'results' / 'benchmark'
    env = environment()
    results = []
    for labels, prob in configs(args.quick, args.seed):
        r = {**labels, 'N': int(prob.Xf.shape[0]), **run_config(prob, args)}
        results.append(r)
        t, s = r['timings'], r['speedup_median']
        print(f"{r['mode']:<8} n={r['n']:4d} K={r['K']} N={r['N']:4d} p={r['p']:3d} M={r['M']:3d}  "
              f"solve obs {t['solve/obs_cholesky']['median_s'] * 1e3:7.3f} ms"
              f"  vs state_cholesky x{s['solve']['state_cholesky']:5.1f}, state_lstsq x{s['solve']['state_lstsq']:6.1f}"
              f"  | full x{s['full']['state_cholesky']:4.1f} / x{s['full']['state_lstsq']:5.1f}"
              f"  | max|diff| {max(r['max_abs_diff'].values()):.1e}")

    protocol = {'problem': f'smooth Gaussian random-field ensemble, correlation length {CORR_LENGTH}; '
                           f'distinct random observed components; R = 0.01 I',
                'smoother': f'K window times {TIME_STEP} apart, members correlated across times; p observations per '
                            f'time with error variance 0.01 (k + 1) at time k; reduced window state of N rows; '
                            f'temporal localization exp(-|t_i - t_j| / {TIME_RADIUS})',
                'localization': f'exp(-|x_i - x_j| / {LOCALIZATION_RADIUS}) on [0, 1)',
                'seed': args.seed, 'threads': args.threads, 'repeats': args.repeats, 'warmup': args.warmup,
                'timer': f'time.perf_counter; stages interleaved; each sample averages calls lasting >= '
                         f'{SAMPLE_SECONDS * 1e3:.0f} ms; garbage collector disabled while timing',
                'cpu_wall_ratio': 'process CPU time / wall time per sample, about 1 for single-threaded runs'}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps({'environment': env, 'protocol': protocol, 'results': results}, indent=2))
    with open(out.with_suffix('.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['mode', 'n', 'K', 'N', 'p', 'M', 'stage', 'method', 'median_s', 'p25_s', 'p75_s', 'min_s',
                    'max_s', 'repeats', 'calls_per_sample', 'cpu_wall_ratio'])
        for r in results:
            for key, s in r['timings'].items():
                stage, method = key.split('/')
                w.writerow([r['mode'], r['n'], r['K'], r['N'], r['p'], r['M'], stage, method, s['median_s'], s['p25_s'],
                            s['p75_s'],
                            s['min_s'], s['max_s'], s['repeats'], s['calls_per_sample'], s['cpu_wall_ratio']])
    print(f'written {out.with_suffix(".json")} and {out.with_suffix(".csv")}')


if __name__ == '__main__':
    main()
