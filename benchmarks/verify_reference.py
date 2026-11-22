"""Numerical comparison of analysis_update with the independent state-space reference solver.

For a grid of state sizes n, observation counts p, ensemble sizes M (including M << n, where the
unlocalized ensemble covariance is rank-deficient), with and without localization and with
diagonal or correlated R, both algorithms process identical inputs and their full analysis
ensembles are compared.

    python benchmarks/verify_reference.py [--out benchmarks/results/verification]

Writes <out>.json and <out>.csv. Errors:
    max_abs_error            max |Xa - Xa_ref|
    max_rel_error            max |Xa - Xa_ref| / max |Xa_ref|
    max_rel_increment_error  max |Xa - Xa_ref| / max |Xa_ref - Xf|
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
from problems import background_covariance, make_problem  # noqa: E402

from ensemble_da import analysis_update, reference_analysis  # noqa: E402

N_VALUES = (80, 160, 320, 640)
P_FRACTIONS = (0.1, 0.2)
M_VALUES = (5, 12, 24, 48)
SEEDS = (0, 1, 2)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', default=str(Path(__file__).resolve().parent / 'results' / 'verification'))
    args = p.parse_args(argv)

    rows = []
    for n, frac, M, localized, correlated, seed in itertools.product(
            N_VALUES, P_FRACTIONS, M_VALUES, (True, False), (False, True), SEEDS):
        pobs = int(round(frac * n))
        prob = make_problem(n, pobs, M, seed, localized=localized, correlated_R=correlated)
        B = background_covariance(prob)
        xa = analysis_update(B, prob.H, prob.Y, prob.R, prob.Xf)
        ref = reference_analysis(B, prob.H, prob.Y, prob.R, prob.Xf)
        diff = np.abs(xa - ref).max()
        rows.append({'n': n, 'p': pobs, 'M': M, 'localized': localized, 'correlated_R': correlated, 'seed': seed,
                     'rank_B': int(np.linalg.matrix_rank(B)),
                     'max_abs_error': float(diff),
                     'max_rel_error': float(diff / np.abs(ref).max()),
                     'max_rel_increment_error': float(diff / np.abs(ref - prob.Xf).max())})

    summary = {key: float(max(r[key] for r in rows)) for key in ('max_abs_error', 'max_rel_error', 'max_rel_increment_error')}
    summary['cases'] = len(rows)
    summary['rank_deficient_cases'] = sum(r['rank_B'] < r['n'] for r in rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps({'summary': summary, 'cases': rows}, indent=2))
    with open(out.with_suffix('.csv'), 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{summary['cases']} cases ({summary['rank_deficient_cases']} with rank-deficient B)")
    for key in ('max_abs_error', 'max_rel_error', 'max_rel_increment_error'):
        print(f'  worst {key:<24} {summary[key]:.2e}')
    print(f'written {out.with_suffix(".json")} and {out.with_suffix(".csv")}')


if __name__ == '__main__':
    main()
