# Verification and benchmarks

Both scripts use seeded synthetic problems from [problems.py](problems.py): a smooth Gaussian
random-field ensemble on n components of [0, 1) (correlation length 0.05), p distinct observed
components, Gaussian observation errors, and localization `exp(-|x_i - x_j| / 0.1)`. Each records
its environment (Python, NumPy, SciPy, BLAS/LAPACK, CPU, OS, thread settings, git commit) through
[environment.py](environment.py).

## Independent numerical reference — `verify_reference.py`

Compares the production update `analysis_update` (Cholesky factorization of the p × p innovation
covariance) with `reference_analysis`, an independent solver of the same minimization: weighted
least squares in state space, with B = S Sᵀ from an eigendecomposition and SVD least squares
(`numpy.linalg.lstsq`). The two share no code.

- **Main grid** (gated): n ∈ {80, 160, 320, 640}, p ∈ {0.1 n, 0.2 n}, M ∈ {5, 12, 24, 48}, with
  and without localization, diagonal and correlated R, three seeds — 384 cases, 192 with a
  rank-deficient (unlocalized, M < n) ensemble covariance. Full analysis ensembles are compared by
  maximum absolute error, error relative to the analysis, and relative to the analysis increment;
  the script exits with code 1 if the worst relative increment error exceeds `--tol` (1e-9).
- **Conditioning slice** (reported): unlocalized B with R = r I, r = 1e-2 … 1e-12. The relative
  error of the observation-space solution grows like machine epsilon × cond(H B Hᵀ + R).

```bash
python benchmarks/verify_reference.py          # -> results/verification.{json,csv}
```

## Timing — `benchmark_analysis.py`

Three solvers of the same analysis on identical inputs:

| Solver | Implementation | Dominant cost |
|---|---|---|
| `obs_cholesky` | `ensemble_da.analysis_update`: Cholesky of H B Hᵀ + R (observation space) | O(p³ + n p M) |
| `state_cholesky` | [baselines.py](baselines.py): information form in state space, n × n Cholesky | O(n³) |
| `state_lstsq` | `ensemble_da.reference_analysis`: state-space least squares (eigh + SVD) | O(n³), larger constant |

`state_cholesky` is the fair baseline for the speed-up of the observation-space formulation;
`state_lstsq` is the verification reference, timed for completeness. Both state-space solvers are
re-implementations of those formulations, not measurements of any other program.

Stages timed separately: covariance estimation, localization (factors and Schur product), solve
(per solver), and full analysis (covariance + localization + solve, per solver). Configurations
(n, p, M): (80, 8, 24), (160, 16, 24), (160, 32, 24), (320, 32, 24), (640, 64, 24), and an
ensemble-size sweep (320, 32, M) with M ∈ {12, 48, 96}.

Protocol: BLAS threads fixed through environment variables before NumPy is imported (default 1);
3 warm-up calls per stage; 30 samples per stage, interleaved across stages so drift affects all
of them alike; each sample averages as many calls as needed to last at least 5 ms (short stages
are not timed one call at a time); `time.perf_counter`, garbage collector off while timing. Each
stage reports median, quartiles, minimum and maximum, and the CPU/wall-time ratio (about 1 for a
single-threaded run). The maximum differences between the solvers' analyses are recorded.

```bash
python benchmarks/benchmark_analysis.py --threads 1 --repeats 30 --warmup 3   # -> results/benchmark.{json,csv}
```

Committed results in [results/](results/) were produced on the machine described in their
`environment` sections.
