"""ensemble_da — model-independent ensemble Kalman analysis (filter and smoother).

The package takes a forecast ensemble, observations, an observation operator and observation
errors, and returns the analysis ensemble. It does not implement a forecast model: sequential
cycles receive the model as a callable.

    from ensemble_da import assimilate, Localization, observation_covariance
"""
from .analysis import analysis_update
from .assimilate import AssimilationResult, assimilate, assimilate_window
from .covariance import Localization, estimate_covariance, great_circle_distance
from .cycle import CycleResult, CycleStep, derive_windows, plan_cycle, run_cycle
from .observations import Observations, observation_covariance, perturbed_observations
from .reference import reference_analysis
from .state import WindowState, build_window_state

__version__ = '0.1.0'

__all__ = [
    '__version__',
    'assimilate', 'assimilate_window', 'AssimilationResult',
    'analysis_update', 'reference_analysis',
    'estimate_covariance', 'Localization', 'great_circle_distance',
    'observation_covariance', 'perturbed_observations', 'Observations',
    'build_window_state', 'WindowState',
    'run_cycle', 'plan_cycle', 'derive_windows', 'CycleResult', 'CycleStep',
]
