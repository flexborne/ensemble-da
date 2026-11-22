"""Command-line interface on .npz arrays."""
from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from ensemble_da import assimilate, cli


def _write_case(path, operator=False):
    rng = np.random.default_rng(0)
    Xf = np.exp(0.2 * rng.standard_normal((10, 6)))
    H = np.array([1, 5, 8])
    arrays = {'forecast': Xf, 'observations': Xf.mean(axis=1)[H] * 1.1, 'observation_covariance': 0.01 * np.eye(3)}
    if operator:                                    # averages of two neighbouring components
        arrays['observation_operator'] = 0.5 * (np.eye(10)[H] + np.eye(10)[H + 1])
        arrays['localization'] = np.exp(-np.abs(np.subtract.outer(np.arange(10), np.arange(10))) / 3.0)
    else:
        arrays['observation_indices'] = H
    np.savez(path, **arrays)
    return arrays


def test_assimilate_command_matches_the_api(tmp_path):
    a = _write_case(tmp_path / 'case.npz')
    assert cli.main(['assimilate', str(tmp_path / 'case.npz'), '-o', str(tmp_path / 'out'), '--no-perturb']) == 0
    expected = assimilate(a['forecast'], a['observations'], a['observation_indices'], a['observation_covariance'], perturb=False)
    assert np.allclose(np.load(tmp_path / 'out.npz')['analysis'], expected.analysis)
    assert json.loads((tmp_path / 'out.json').read_text())['n_obs'] == 3


def test_operator_localization_transform_and_seed_reach_the_api(tmp_path):
    a = _write_case(tmp_path / 'case.npz', operator=True)
    assert cli.main(['assimilate', str(tmp_path / 'case.npz'), '-o', str(tmp_path / 'out'),
                     '--transform', 'log', '--seed', '3']) == 0
    expected = assimilate(a['forecast'], a['observations'], a['observation_operator'], a['observation_covariance'],
                          localization=a['localization'], transform='log', rng=3)
    assert np.allclose(np.load(tmp_path / 'out.npz')['analysis'], expected.analysis)


def test_dotted_output_names_stay_distinct_and_the_input_is_protected(tmp_path):
    _write_case(tmp_path / 'case.npz')
    for name in ('member.1', 'member.2'):
        assert cli.main(['assimilate', str(tmp_path / 'case.npz'), '-o', str(tmp_path / name)]) == 0
    assert (tmp_path / 'member.1.npz').exists() and (tmp_path / 'member.2.npz').exists()
    assert cli.main(['assimilate', str(tmp_path / 'case.npz'), '-o', str(tmp_path / 'case')]) == 1
    assert 'forecast' in np.load(tmp_path / 'case.npz')


def test_missing_operator_is_a_clean_error(tmp_path, capsys):
    np.savez(tmp_path / 'bad.npz', forecast=np.zeros((3, 2)), observations=np.zeros(1), observation_covariance=np.eye(1))
    assert cli.main(['assimilate', str(tmp_path / 'bad.npz'), '-o', str(tmp_path / 'out')]) == 1
    assert 'observation_indices' in capsys.readouterr().err


def test_version_from_an_unrelated_directory(tmp_path):
    out = subprocess.run([sys.executable, '-m', 'ensemble_da', '--version'], cwd=tmp_path, capture_output=True, text=True)
    assert out.returncode == 0 and 'ensemble-da' in out.stdout
