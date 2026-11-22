"""Timing of the analysis step: observation-space Cholesky vs state-space least squares.

Both algorithms solve the same minimization on identical, seeded inputs:
  cholesky  ensemble_da.analysis_update     - Cholesky of the p x p innovation covariance
  lstsq     ensemble_da.reference_analysis  - weighted least squares in state space
            (eigendecomposition of B + SVD least squares, the equivalent regression formulation)
The lstsq solver is a re-implementation of that formulation, not a measurement of any other program.

Stages, each timed separately: covariance estimation, localization (factors + Schur product),
solve (per algorithm), and the full analysis (covariance + localization + solve).

    python benchmarks/benchmark_analysis.py [--threads 1] [--repeats 30] [--warmup 3] [--out PATH]

BLAS threads are fixed through environment variables set before NumPy is imported. Writes
<out>.json (environment + all statistics) and <out>.csv (one row per configuration/stage/method).
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
    p.add_argument('--repeats', type=int, default=30, help='timed repetitions per stage (default 30)')
    p.add_argument('--warmup', type=int, default=3, help='untimed warm-up calls per stage (default 3)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None, help='output path without extension')
    return p.parse_args(argv)


ARGS = _parse(sys.argv[1:] if __name__ == '__main__' else [])
for _var in THREAD_VARS:                       # must happen before NumPy loads its BLAS
    os.environ[_var] = str(ARGS.threads)

import csv  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from problems import CORR_LENGTH, LOCALIZATION_RADIUS, make_problem  # noqa: E402

import ensemble_da  # noqa: E402
from ensemble_da import analysis_update, estimate_covariance, reference_analysis  # noqa: E402

CONFIGS = [  # (n, p, M)
    (80, 8, 24), (160, 16, 24), (160, 32, 24), (320, 32, 24), (640, 64, 24),
    (320, 32, 12), (320, 32, 48), (320, 32, 96),      # ensemble-size sweep
]


def _cpu_model():
    try:
        if sys.platform == 'darwin':
            return subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True,
                                  text=True, check=True).stdout.strip()
        if sys.platform.startswith('linux'):
            for line in Path('/proc/cpuinfo').read_text().splitlines():
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or 'unknown'


def _blas_info():
    info = {}
    for name, mod in (('numpy', np), ('scipy', scipy)):
        try:
            deps = mod.show_config(mode='dicts').get('Build Dependencies', {})
            info[name] = {k: {f: deps[k].get(f) for f in ('name', 'version')} for k in ('blas', 'lapack') if k in deps}
        except Exception as exc:              # older releases without mode='dicts'
            info[name] = f'unavailable: {exc}'
    try:
        import threadpoolctl
        info['threadpoolctl'] = threadpoolctl.threadpool_info()
    except ImportError:
        info['threadpoolctl'] = 'not installed'
    return info


def _git_commit():
    """HEAD commit, suffixed '-dirty' when the working tree has uncommitted changes."""
    here = Path(__file__).resolve().parent
    try:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=here, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=here,
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ('-dirty' if dirty else '')
    except (OSError, subprocess.SubprocessError):
        return None


def environment():
    return {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'python': platform.python_version(), 'implementation': platform.python_implementation(),
        'os': f'{platform.system()} {platform.release()}', 'machine': platform.machine(),
        'cpu': _cpu_model(), 'logical_cpus': os.cpu_count(),
        'numpy': np.__version__, 'scipy': scipy.__version__, 'ensemble_da': ensemble_da.__version__,
        'blas': _blas_info(), 'thread_env': {v: os.environ.get(v) for v in THREAD_VARS},
        'git_commit': _git_commit(),
    }


def _time(fn, repeats, warmup):
    for _ in range(warmup):
        fn()
    samples = []
    gc.collect()
    gc.disable()
    try:
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - t0)
    finally:
        gc.enable()
    s = np.asarray(samples)
    return {'median_s': float(np.median(s)), 'p25_s': float(np.percentile(s, 25)),
            'p75_s': float(np.percentile(s, 75)), 'min_s': float(s.min()), 'max_s': float(s.max()),
            'repeats': repeats}


def run_config(n, p, M, args):
    prob = make_problem(n, p, M, args.seed)
    C = estimate_covariance(prob.Xf)
    B = C * prob.localization.matrix()
    xa_chol = analysis_update(B, prob.H, prob.Y, prob.R, prob.Xf)
    xa_lsq = reference_analysis(B, prob.H, prob.Y, prob.R, prob.Xf)
    diff = np.abs(xa_chol - xa_lsq).max()

    def full(solver):
        Bf = estimate_covariance(prob.Xf) * prob.localization.matrix()
        return solver(Bf, prob.H, prob.Y, prob.R, prob.Xf)

    stages = {
        ('covariance', 'shared'): lambda: estimate_covariance(prob.Xf),
        ('localization', 'shared'): lambda: C * prob.localization.matrix(),
        ('solve', 'cholesky'): lambda: analysis_update(B, prob.H, prob.Y, prob.R, prob.Xf),
        ('solve', 'lstsq'): lambda: reference_analysis(B, prob.H, prob.Y, prob.R, prob.Xf),
        ('full', 'cholesky'): lambda: full(analysis_update),
        ('full', 'lstsq'): lambda: full(reference_analysis),
    }
    timings = {f'{stage}/{method}': _time(fn, args.repeats, args.warmup) for (stage, method), fn in stages.items()}
    speedup = {s: timings[f'{s}/lstsq']['median_s'] / timings[f'{s}/cholesky']['median_s'] for s in ('solve', 'full')}
    return {'n': n, 'p': p, 'M': M, 'timings': timings, 'speedup_median': speedup,
            'max_abs_diff': float(diff), 'max_rel_diff': float(diff / np.abs(xa_lsq).max())}


def main():
    args = ARGS
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / 'results' / 'benchmark'
    env = environment()
    results = []
    for n, p, M in CONFIGS:
        r = run_config(n, p, M, args)
        results.append(r)
        t = r['timings']
        print(f"n={n:4d} p={p:3d} M={M:3d}  solve: cholesky {t['solve/cholesky']['median_s'] * 1e3:8.3f} ms"
              f"  lstsq {t['solve/lstsq']['median_s'] * 1e3:9.3f} ms  (x{r['speedup_median']['solve']:6.1f})"
              f"  full x{r['speedup_median']['full']:5.1f}  max|diff| {r['max_abs_diff']:.1e}")

    method = {'problem': f'smooth Gaussian random-field ensemble, correlation length {CORR_LENGTH}; '
                         f'distinct random observed components; R = 0.01 I',
              'localization': f'exp(-|x_i - x_j| / {LOCALIZATION_RADIUS}) on [0, 1)',
              'seed': args.seed, 'threads': args.threads, 'repeats': args.repeats, 'warmup': args.warmup,
              'timer': 'time.perf_counter, garbage collector disabled while timing'}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps({'environment': env, 'method': method, 'results': results}, indent=2))
    with open(out.with_suffix('.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['n', 'p', 'M', 'stage', 'method', 'median_s', 'p25_s', 'p75_s', 'min_s', 'max_s', 'repeats',
                    'max_abs_diff', 'max_rel_diff'])
        for r in results:
            for key, s in r['timings'].items():
                stage, meth = key.split('/')
                w.writerow([r['n'], r['p'], r['M'], stage, meth, s['median_s'], s['p25_s'], s['p75_s'],
                            s['min_s'], s['max_s'], s['repeats'], r['max_abs_diff'], r['max_rel_diff']])
    print(f'written {out.with_suffix(".json")} and {out.with_suffix(".csv")}')


if __name__ == '__main__':
    main()
