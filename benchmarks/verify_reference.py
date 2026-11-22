"""Numerical comparison of analysis_update with the independent state-space reference solver.

Main grid: state sizes n, observation counts p, ensemble sizes M (including M << n, where the
unlocalized ensemble covariance is rank-deficient), with and without localization, diagonal or
correlated R; both solvers process identical inputs and their full analysis ensembles are compared.
The run fails (exit code 1) if the worst error relative to the increment exceeds --tol.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from environment import environment  # noqa: E402
from problems import background_covariance, make_problem  # noqa: E402

from ensemble_da import analysis_update, reference_analysis  # noqa: E402

N_VALUES = (80, 160, 320, 640)
P_FRACTIONS = (0.1, 0.2)
M_VALUES = (5, 12, 24, 48)
SEEDS = (0, 1, 2)
CONDITIONING_R = (1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12)


def compare(prob) -> dict:
    B = background_covariance(prob)
    xa = analysis_update(B, prob.H, prob.Y, prob.R, prob.Xf)
    ref = reference_analysis(B, prob.H, prob.Y, prob.R, prob.Xf)
    diff = np.abs(xa - ref).max()
    return {'rank_B': int(np.linalg.matrix_rank(B)),
            'cond_S': float(np.linalg.cond(B[np.ix_(prob.H, prob.H)] + prob.R)),
            'max_abs_error': float(diff),
            'max_rel_error': float(diff / np.abs(ref).max()),
            'max_rel_increment_error': float(diff / np.abs(ref - prob.Xf).max())}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--tol', type=float, default=1e-9, help='limit on the worst max_rel_increment_error')
    p.add_argument('--out', default=str(Path(__file__).resolve().parent / 'results' / 'verification'))
    args = p.parse_args(argv)

    rows = []
    for n, frac, M, localized, correlated, seed in itertools.product(
            N_VALUES, P_FRACTIONS, M_VALUES, (True, False), (False, True), SEEDS):
        pobs = int(round(frac * n))
        rows.append({'n': n, 'p': pobs, 'M': M, 'localized': localized, 'correlated_R': correlated, 'seed': seed,
                     **compare(make_problem(n, pobs, M, seed, localized=localized, correlated_R=correlated))})
    conditioning = [{'n': 200, 'p': 60, 'M': 10, 'obs_variance': r,
                     **compare(make_problem(200, 60, 10, 0, localized=False, obs_variance=r))} for r in CONDITIONING_R]

    keys = ('max_abs_error', 'max_rel_error', 'max_rel_increment_error')
    summary = {key: float(max(r[key] for r in rows)) for key in keys}
    summary.update(cases=len(rows), rank_deficient_cases=sum(r['rank_B'] < r['n'] for r in rows),
                   max_cond_S=float(max(r['cond_S'] for r in rows)), tol=args.tol,
                   passed=summary['max_rel_increment_error'] <= args.tol)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps(
        {'environment': environment(), 'summary': summary, 'cases': rows, 'conditioning': conditioning}, indent=2))
    with open(out.with_suffix('.csv'), 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{summary['cases']} cases ({summary['rank_deficient_cases']} with rank-deficient B), "
          f"cond(H B H^T + R) up to {summary['max_cond_S']:.1e}")
    for key in keys:
        print(f'  worst {key:<24} {summary[key]:.2e}')
    print('conditioning slice (unlocalized B, R = r I):')
    for c in conditioning:
        print(f"  r = {c['obs_variance']:.0e}  cond {c['cond_S']:.1e}  rel increment error {c['max_rel_increment_error']:.1e}")
    print(f'written {out.with_suffix(".json")} and {out.with_suffix(".csv")}')
    if not summary['passed']:
        print(f"FAILED: worst relative increment error {summary['max_rel_increment_error']:.2e} > {args.tol:.0e}")
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
