"""Command-line interface: one analysis step on arrays stored in an .npz file.

Input keys: forecast (n, M), observations (p,), observation_covariance (p, p), and either
observation_indices (p,) or observation_operator (p, n); optional localization (n, n).
Output: <output>.npz with `analysis` (n, M) and <output>.json with the diagnostics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .assimilate import TRANSFORMS, assimilate


def _cmd_assimilate(args) -> int:
    with np.load(args.input) as data:
        if 'observation_indices' in data:
            H = data['observation_indices']
        elif 'observation_operator' in data:
            H = data['observation_operator']
        else:
            raise KeyError('input needs observation_indices or observation_operator')
        result = assimilate(data['forecast'], data['observations'], H, data['observation_covariance'],
                            localization=data['localization'] if 'localization' in data else None,
                            transform=args.transform, perturb=not args.no_perturb, rng=args.seed)
    base = Path(args.output.removesuffix('.npz').removesuffix('.json'))
    npz, js = base.parent / f'{base.name}.npz', base.parent / f'{base.name}.json'
    if npz.resolve() == Path(args.input).resolve():
        raise ValueError(f'output {npz} would overwrite the input')
    np.savez(npz, analysis=result.analysis)
    js.write_text(json.dumps(result.diagnostics, indent=2))
    print(f'analysis written to {npz} (diagnostics: {js})')
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='ensemble-da', description='Ensemble Kalman analysis on NumPy arrays')
    p.add_argument('--version', action='version', version=f'ensemble-da {__version__}')
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('assimilate', help='one analysis step: input .npz -> output .npz + .json')
    a.add_argument('input', help='input .npz')
    a.add_argument('-o', '--output', required=True, help='output path (.npz and .json are written)')
    a.add_argument('--transform', choices=TRANSFORMS, default='none', help='analysis space (default: none)')
    a.add_argument('--no-perturb', action='store_true', help='deterministic: no observation perturbations')
    a.add_argument('--seed', type=int, default=None, help='seed for observation perturbations')
    a.set_defaults(func=_cmd_assimilate)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError, OSError, np.linalg.LinAlgError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
