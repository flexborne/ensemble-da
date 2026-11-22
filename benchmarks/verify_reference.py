"""Numerical comparison of the analysis with the independent state-space reference solver.

Filter grid: analysis_update on state sizes n, observation counts p, ensemble sizes M (including
M << n, where the unlocalized ensemble covariance is rank-deficient), with and without
localization, diagonal or correlated R.
Smoother grid: assimilate_window on K-time windows (reduced window state, temporal localization)
against the reference solved on the full multi-time state of every window time.
Both solvers process identical inputs and their full analysis ensembles are compared; the run fails
(exit code 1) if the worst error relative to the increment exceeds --tol in either grid.

Conditioning slice (reported, not gated): unlocalized B with shrinking R = r I, showing that the
observation-space solution loses about eps * cond(H B H^T + R) of relative accuracy.

    python benchmarks/verify_reference.py [--tol 1e-9] [--out benchmarks/results/verification]

Errors:  max_abs_error = max |Xa - Xa_ref|;  max_rel_error = max_abs_error / max |Xa_ref|;
         max_rel_increment_error = max_abs_error / max |Xa_ref - Xf|.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag

sys.path.insert(0, str(Path(__file__).resolve().parent))
from environment import environment  # noqa: E402
from problems import background_covariance, make_problem, make_window_problem  # noqa: E402

from ensemble_da import (analysis_update, assimilate_window, estimate_covariance,  # noqa: E402
                         perturbed_observations, reference_analysis)

N_VALUES = (80, 160, 320, 640)
P_FRACTIONS = (0.1, 0.2)
M_VALUES = (5, 12, 24, 48)
SEEDS = (0, 1, 2)
WINDOW_N, WINDOW_K, WINDOW_P_FRACTION = (80, 160, 320), (2, 3), 0.1    # p per window time
CONDITIONING_R = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)
KEYS = ('max_abs_error', 'max_rel_error', 'max_rel_increment_error')


def _errors(xa, ref, xf, B, H, R) -> dict:
    diff = np.abs(xa - ref).max()
    return {'rank_B': int(np.linalg.matrix_rank(B)),
            'cond_S': float(np.linalg.cond(B[np.ix_(H, H)] + R)),
            'max_abs_error': float(diff),
            'max_rel_error': float(diff / np.abs(ref).max()),
            'max_rel_increment_error': float(diff / np.abs(ref - xf).max())}


def compare(prob) -> dict:
    B = background_covariance(prob)
    xa = analysis_update(B, prob.H, prob.Y, prob.R, prob.Xf)
    ref = reference_analysis(B, prob.H, prob.Y, prob.R, prob.Xf)
    return _errors(xa, ref, prob.Xf, B, prob.H, prob.R)


def compare_window(wp, seed) -> dict:
    """assimilate_window against the reference on the full state of every window time."""
    K, (n, M) = len(wp.ensembles), wp.ensembles[-1].shape
    xa = assimilate_window(wp.ensembles, wp.observations, wp.times, localization=wp.localization, rng=seed).analysis
    Z = np.vstack(wp.ensembles)                                       # row k n + c: component c at time k
    B = estimate_covariance(Z)
    if wp.localization is not None:
        B = B * wp.localization.matrix(np.tile(np.arange(n), K), np.repeat(wp.times, n))
    H = np.concatenate([k * n + o.indices for k, o in enumerate(wp.observations)])
    R = block_diag(*[o.covariance for o in wp.observations])
    Y = perturbed_observations(np.concatenate([o.values for o in wp.observations]), R, M, np.random.default_rng(seed))
    ref = reference_analysis(B, H, Y, R, Z)[-n:]
    return _errors(xa, ref, wp.ensembles[-1], B, H, R)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--tol', type=float, default=1e-9, help='limit on the worst max_rel_increment_error')
    p.add_argument('--out', default=str(Path(__file__).resolve().parent / 'results' / 'verification'))
    args = p.parse_args(argv)

    rows = []
    for n, frac, M, localized, correlated, seed in itertools.product(
            N_VALUES, P_FRACTIONS, M_VALUES, (True, False), (False, True), SEEDS):
        pobs = int(round(frac * n))
        rows.append({'mode': 'filter', 'n': n, 'K': 1, 'p': pobs, 'M': M, 'localized': localized,
                     'correlated_R': correlated, 'seed': seed,
                     **compare(make_problem(n, pobs, M, seed, localized=localized, correlated_R=correlated))})
    for n, K, M, localized, correlated, seed in itertools.product(
            WINDOW_N, WINDOW_K, M_VALUES, (True, False), (False, True), SEEDS):
        pobs = int(round(WINDOW_P_FRACTION * n))
        wp = make_window_problem(n, pobs, M, K, seed, localized=localized, correlated_R=correlated)
        rows.append({'mode': 'smoother', 'n': n, 'K': K, 'p': pobs * K, 'M': M, 'localized': localized,
                     'correlated_R': correlated, 'seed': seed, **compare_window(wp, seed)})
    conditioning = [{'n': 200, 'p': 60, 'M': 10, 'obs_variance': r,
                     **compare(make_problem(200, 60, 10, 0, localized=False, obs_variance=r))} for r in CONDITIONING_R]

    summary = {'tol': args.tol}
    for mode in ('filter', 'smoother'):
        rs = [r for r in rows if r['mode'] == mode]
        summary[mode] = {**{key: float(max(r[key] for r in rs)) for key in KEYS},
                         'cases': len(rs), 'rank_deficient_cases': sum(r['rank_B'] < r['K'] * r['n'] for r in rs),
                         'max_cond_S': float(max(r['cond_S'] for r in rs))}
    summary['passed'] = all(summary[m]['max_rel_increment_error'] <= args.tol for m in ('filter', 'smoother'))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps(
        {'environment': environment(), 'summary': summary, 'cases': rows, 'conditioning': conditioning}, indent=2))
    with open(out.with_suffix('.csv'), 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for mode in ('filter', 'smoother'):
        s = summary[mode]
        print(f"{mode}: {s['cases']} cases ({s['rank_deficient_cases']} with rank-deficient B), "
              f"cond(H B H^T + R) up to {s['max_cond_S']:.1e}")
        for key in KEYS:
            print(f'  worst {key:<24} {s[key]:.2e}')
    print('conditioning slice (unlocalized B, R = r I):')
    for c in conditioning:
        print(f"  r = {c['obs_variance']:.0e}  cond {c['cond_S']:.1e}  rel increment error {c['max_rel_increment_error']:.1e}")
    print(f'written {out.with_suffix(".json")} and {out.with_suffix(".csv")}')
    if not summary['passed']:
        worst = max(summary[m]['max_rel_increment_error'] for m in ('filter', 'smoother'))
        print(f'FAILED: worst relative increment error {worst:.2e} > {args.tol:.0e}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
