# Verification and benchmarks

Two scripts, both on seeded synthetic problems from [problems.py](problems.py): a smooth Gaussian
random-field ensemble on n components of [0, 1) (correlation length 0.05), p distinct observed
components, Gaussian observation errors, and localization `exp(-|x_i - x_j| / 0.1)`.

## Independent numerical reference — `verify_reference.py`

Compares the production update `analysis_update` (Cholesky factorization of the p × p innovation
covariance) with `reference_analysis`, an independent solver of the same minimization: weighted
least squares in state space, with B = S Sᵀ from an eigendecomposition and SVD least squares
(`numpy.linalg.lstsq`). The two share no code.

Grid: n ∈ {80, 160, 320, 640}, p ∈ {0.1 n, 0.2 n}, M ∈ {5, 12, 24, 48}, with and without
localization, diagonal and correlated R, three seeds — 384 cases, 192 of them with a rank-deficient
(unlocalized, M < n) ensemble covariance. Full analysis ensembles are compared by maximum absolute
error, maximum error relative to the analysis, and relative to the analysis increment.

```bash
python benchmarks/verify_reference.py          # -> results/verification.{json,csv}
```

## Timing — `benchmark_analysis.py`

Times, on identical inputs, the two formulations of the analysis:

| Method | Implementation | Dominant cost |
|---|---|---|
| `cholesky` | `analysis_update`: Cholesky of H B Hᵀ + R | O(p³ + n p M) |
| `lstsq` | `reference_analysis`: state-space weighted least squares | O(n³) |

`lstsq` is a re-implementation of the equivalent regression formulation; it is not a measurement
of any other program. Stages timed separately: covariance estimation, localization (factors and
Schur product), solve (per method), full analysis (covariance + localization + solve).

Configurations (n, p, M): (80, 8, 24), (160, 16, 24), (160, 32, 24), (320, 32, 24),
(640, 64, 24), and an ensemble-size sweep (320, 32, M) with M ∈ {12, 48, 96}.

Protocol: BLAS threads fixed through environment variables before NumPy is imported (default 1);
3 untimed warm-up calls and 30 timed calls per stage with `time.perf_counter`, garbage collector
disabled while timing; median, quartiles, minimum and maximum reported, plus the maximum absolute
and relative difference between the two methods' analyses. The JSON records Python, NumPy, SciPy,
BLAS/LAPACK, CPU model, OS, thread settings and the git commit.

```bash
python benchmarks/benchmark_analysis.py --threads 1 --repeats 30 --warmup 3   # -> results/benchmark.{json,csv}
```

Committed results in [results/](results/) were produced on the machine described in
`results/benchmark.json` (`environment`).
