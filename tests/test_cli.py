"""Command-line interface on .npz arrays."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from ensemble_da import assimilate, cli


def _write_case(path, operator_key='observation_indices'):
    rng = np.random.default_rng(0)
    Xf = rng.standard_normal((10, 6))
    H = np.array([1, 5, 8])
    arrays = {'forecast': Xf, 'observations': Xf.mean(axis=1)[H] + 0.2, 'observation_covariance': 0.1 * np.eye(3),
              operator_key: H if operator_key == 'observation_indices' else np.eye(10)[H]}
    np.savez(path, **arrays)
    return Xf, arrays['observations'], H


def test_assimilate_command_matches_the_api(tmp_path):
    Xf, y, H = _write_case(tmp_path / 'case.npz')
    assert cli.main(['assimilate', str(tmp_path / 'case.npz'), '-o', str(tmp_path / 'out'), '--no-perturb']) == 0
    expected = assimilate(Xf, y, H, 0.1 * np.eye(3), perturb=False)
    assert np.allclose(np.load(tmp_path / 'out.npz')['analysis'], expected.analysis)
    assert json.loads((tmp_path / 'out.json').read_text())['n_obs'] == 3


def test_matrix_operator_input(tmp_path):
    _write_case(tmp_path / 'case.npz', operator_key='observation_operator')
    assert cli.main(['assimilate', str(tmp_path / 'case.npz'), '-o', str(tmp_path / 'out'), '--seed', '1']) == 0


def test_missing_operator_is_a_clean_error(tmp_path, capsys):
    np.savez(tmp_path / 'bad.npz', forecast=np.zeros((3, 2)), observations=np.zeros(1), observation_covariance=np.eye(1))
    assert cli.main(['assimilate', str(tmp_path / 'bad.npz'), '-o', str(tmp_path / 'out')]) == 1
    assert 'observation_indices' in capsys.readouterr().err


def test_version_from_an_unrelated_directory(tmp_path):
    out = subprocess.run([sys.executable, '-m', 'ensemble_da', '--version'], cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0 and 'ensemble-da' in out.stdout
