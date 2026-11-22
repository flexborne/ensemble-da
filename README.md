# ensemble-da

Model-independent ensemble Kalman analysis: filter and smoother with distance-based covariance
localization, for any forecast model that can produce an ensemble.

The package receives a forecast ensemble, observations, an observation operator and observation
errors, and returns the analysis ensemble with diagnostics. It does not implement a forecast
model; a sequential assimilation-forecast cycle takes the model as a function.

> Українська версія: [README.uk.md](README.uk.md)

---

## The analysis problem

For each ensemble member m the analysis minimizes

$$J(x) = (x - x^f_m)^\top B^{-1} (x - x^f_m) + (H x - y_m)^\top R^{-1} (H x - y_m),$$

whose minimizer is

$$X^a = X^f + B H^\top (H B H^\top + R)^{-1} (Y - H X^f).$$

- **Background covariance.** $B = \rho \circ C$, where $C$ is the sample covariance of the forecast
  ensemble $X^f$ (n × M) and $\rho$ the localization factors (Schur product). Without localization
  $B = C$ has rank at most M − 1; the update is still well defined.
- **Solver.** One Cholesky factorization of the p × p matrix $H B H^\top + R$: the cost grows with
  the number of observations p, not cubically with the state size n.
- **Localization.** $\rho_{ij} = e^{-d_h(i,j)/L_h}\, e^{-d_v(i,j)/L_v}\, e^{-|t_i - t_j|/L_t}$ from
  per-component coordinates (Euclidean or great-circle horizontal distance, vertical distance, time).
  Selected components can be decoupled from all others.
- **Observation errors.** $R$ is any symmetric positive-definite matrix; the helper builds
  $R = \mathrm{diag}(\sigma_\text{meas}^2 + \sigma_\text{repr}^2)$ from separate measurement and
  representativeness errors.
- **Stochastic EnKF.** Each member assimilates $y_m = y + \varepsilon_m$, $\varepsilon_m \sim N(0, R)$
  (can be switched off).
- **Log-normal quantities.** With `transform='log'` positive states and observations are analysed
  as logarithms (R is then the covariance of $\ln y$, e.g. $\sigma = \ln \mathrm{GSD}$) and returned
  as positive values.
- **Smoother windows.** An analysis at time $t_K$ may use observations from neighbouring times
  $t_1 \dots t_K$. The window state stacks the observed components of the earlier times and the
  full state at $t_K$; covariances across times come from the ensemble. This reduced state gives
  exactly the same analysis at $t_K$ as the full multi-time state.

## Inputs and outputs

| | Object | Shape |
|---|---|---|
| in | forecast ensemble $X^f$ | (n, M), M ≥ 2 |
| in | observations $y$ | (p,) |
| in | observation operator $H$ | (p,) observed component indices, or a (p, n) matrix |
| in | observation-error covariance $R$ | (p, p) |
| in | localization $\rho$ (optional) | (n, n), from per-component coordinates |
| out | analysis ensemble $X^a$ | (n, M) — restart states for the next forecast |
| out | diagnostics | innovation and residual RMS, forecast and analysis spread, sizes |

## Install

```bash
pip install .                 # numpy and scipy only
pip install ".[test,plot]"    # + pytest, matplotlib for the example figure
```

Python 3.11–3.13.

## Minimal example

```python
import numpy as np
from ensemble_da import assimilate, Localization, observation_covariance

rng = np.random.default_rng(0)
n, M = 100, 20
x = np.linspace(0, 1, n)
Xf = np.cumsum(rng.standard_normal((n, M)), axis=0) / 10      # forecast ensemble from any model
H = np.arange(0, n, 5)                                         # observed components
y = rng.standard_normal(H.size)                                # observations
R = observation_covariance(measurement_std=0.1, representativeness_std=0.05, n_obs=H.size)
rho = Localization(horizontal=x[:, None], horizontal_radius=0.1).matrix()

result = assimilate(Xf, y, H, R, localization=rho, rng=1)
result.analysis        # (n, M) analysis ensemble
result.diagnostics     # {'innovation_rms': ..., 'residual_rms': ..., ...}
```

## Sequential cycle with an external model

```python
from ensemble_da import Observations, run_cycle

def forecast(ensemble, t_from, t_to):          # your model: advance every member
    ...
    return ensemble_at_t_to

observations = {t: Observations(values, indices, R) for t, values in ...}
out = run_cycle(forecast, initial_ensemble, times, observations,
                mode='smoother', window=0.25, localization=Localization(...), rng=1)
out.analyses[t], out.forecasts[t], out.steps
```

Modes: `noda` (forecast only), `filter` (analyse each time with its own observations), `smoother`
(analyse each time with the observations within ± `window`). Every time but the last is analysed;
each analysis seeds the forecast to the next time, and the smoother keeps forecasting ahead to the
end of the next window so its future states exist. `plan_cycle` returns the plan without running it.

## Command line

```bash
ensemble-da assimilate case.npz -o result [--transform log] [--no-perturb] [--seed 1]
```

`case.npz` holds `forecast`, `observations`, `observation_covariance`, `observation_indices` (or
`observation_operator`) and optionally `localization`; the analysis is written to `result.npz`, the
diagnostics to `result.json`.

## Reproducing the experiments

```bash
pytest -q                                      # test suite, no external data
python examples/demo_transport_1d.py --plot demo.png
python benchmarks/verify_reference.py          # comparison with an independent solver
python benchmarks/benchmark_analysis.py        # timing; see benchmarks/README.md
```

- **Twin experiment** ([examples/demo_transport_1d.py](examples/demo_transport_1d.py)): a 1-D
  advection-diffusion model (200 cells, 20 members, 50 log-normal observations, uncertain
  velocity) generates the forecast ensemble and synthetic observations of a known truth. It reports
  the forecast and analysis errors of one analysis (with and without localization) and of
  noda/filter/smoother cycles. The toy model lives in `examples/`, outside the package.
- **Independent reference** ([benchmarks/verify_reference.py](benchmarks/verify_reference.py)):
  the production solver against state-space weighted least squares on 384 problems, including
  rank-deficient ensemble covariances.
- **Benchmarks** ([benchmarks/benchmark_analysis.py](benchmarks/benchmark_analysis.py)): stage
  timings of both formulations with the environment recorded; raw results in
  [benchmarks/results/](benchmarks/results/).

## Package layout

| Module | Contents |
|---|---|
| `assimilate.py` | `assimilate`, `assimilate_window`, `AssimilationResult` — the public analysis step |
| `analysis.py` | `analysis_update` — observation-space Kalman update (Cholesky) |
| `covariance.py` | `estimate_covariance`, `Localization`, `great_circle_distance` |
| `observations.py` | `observation_covariance`, `perturbed_observations`, `Observations` |
| `state.py` | `build_window_state` — reduced multi-time state of a smoother window |
| `transform.py` | guarded log / exp for log-normal quantities |
| `cycle.py` | `plan_cycle`, `run_cycle` — sequential cycle with an injected forecast function |
| `reference.py` | `reference_analysis` — independent solver for verification and benchmarks |
| `cli.py` | `ensemble-da assimilate` |

## License

MIT — see [LICENSE](LICENSE). Citation metadata: [CITATION.cff](CITATION.cff).
