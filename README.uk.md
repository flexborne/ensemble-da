# ensemble-da

Модельно-незалежний ансамблевий аналіз Калмана: фільтр і згладжувач із локалізацією коваріацій за
відстанню — для будь-якої прогнозної моделі, що може видати ансамбль.

Пакет отримує прогнозний ансамбль, спостереження, оператор спостережень і похибки спостережень та
повертає ансамбль аналізу з діагностикою. Прогнозну модель пакет не реалізує: послідовний цикл
«аналіз–прогноз» приймає модель як функцію.

> English version: [README.md](README.md)

---

## Задача аналізу

Для кожного члена ансамблю m аналіз мінімізує

$$J(x) = (x - x^f_m)^\top B^{-1} (x - x^f_m) + (H x - y_m)^\top R^{-1} (H x - y_m),$$

мінімізатор якого

$$X^a = X^f + B H^\top (H B H^\top + R)^{-1} (Y - H X^f).$$

- **Фонова коваріація.** $B = \rho \circ C$, де $C$ — вибіркова коваріація прогнозного ансамблю
  $X^f$ (n × M), а $\rho$ — коефіцієнти локалізації (добуток Шура). Без локалізації $B = C$ має ранг
  не більше M − 1; оновлення при цьому коректно визначене.
- **Розв'язувач.** Один розклад Холецького матриці $H B H^\top + R$ розміру p × p: вартість залежить
  від кількості спостережень p, а не кубічно від розмірності стану n.
- **Локалізація.** $\rho_{ij} = e^{-d_h(i,j)/L_h}\, e^{-d_v(i,j)/L_v}\, e^{-|t_i - t_j|/L_t}$ за
  координатами компонентів стану (евклідова або ортодромічна горизонтальна відстань, вертикальна
  відстань, час). Окремі компоненти можна повністю відв'язати від інших.
- **Похибки спостережень.** $R$ — довільна симетрична додатно визначена матриця; допоміжна функція
  будує $R = \mathrm{diag}(\sigma_\text{вим}^2 + \sigma_\text{репр}^2)$ з окремих похибок вимірювань
  і репрезентативності.
- **Стохастичний EnKF.** Кожен член асимілює $y_m = y + \varepsilon_m$, $\varepsilon_m \sim N(0, R)$
  (можна вимкнути).
- **Логнормальні величини.** З `transform='log'` додатні стани й спостереження аналізуються в
  логарифмах (тоді $R$ — коваріація $\ln y$, напр. $\sigma = \ln \mathrm{GSD}$) і повертаються
  додатними.
- **Вікна згладжувача.** Аналіз у момент $t_K$ може використовувати спостереження сусідніх моментів
  $t_1 \dots t_K$. Стан вікна складається зі спостережуваних компонентів попередніх моментів і
  повного стану в $t_K$; коваріації між моментами беруться з ансамблю. Такий скорочений стан дає
  в $t_K$ точно той самий аналіз, що й повний багаточасовий стан.

## Входи та виходи

| | Об'єкт | Розмір |
|---|---|---|
| вхід | прогнозний ансамбль $X^f$ | (n, M), M ≥ 2 |
| вхід | спостереження $y$ | (p,) |
| вхід | оператор спостережень $H$ | (p,) індекси спостережуваних компонентів або матриця (p, n) |
| вхід | коваріація похибок спостережень $R$ | (p, p) |
| вхід | локалізація $\rho$ (необов'язково) | (n, n), за координатами компонентів |
| вихід | ансамбль аналізу $X^a$ | (n, M) — початкові стани для наступного прогнозу |
| вихід | діагностика | RMS нев'язок до й після аналізу, розкид прогнозу й аналізу, розміри |

## Встановлення

```bash
pip install .                 # лише numpy і scipy
pip install ".[test,plot]"    # + pytest, matplotlib для рисунка прикладу
```

Python 3.11–3.13.

## Мінімальний приклад

```python
import numpy as np
from ensemble_da import assimilate, Localization, observation_covariance

rng = np.random.default_rng(0)
n, M = 100, 20
x = np.linspace(0, 1, n)
Xf = np.cumsum(rng.standard_normal((n, M)), axis=0) / 10      # прогнозний ансамбль будь-якої моделі
H = np.arange(0, n, 5)                                         # спостережувані компоненти
y = rng.standard_normal(H.size)                                # спостереження
R = observation_covariance(measurement_std=0.1, representativeness_std=0.05, n_obs=H.size)
rho = Localization(horizontal=x[:, None], horizontal_radius=0.1).matrix()

result = assimilate(Xf, y, H, R, localization=rho, rng=1)
result.analysis        # (n, M) ансамбль аналізу
result.diagnostics     # {'innovation_rms': ..., 'residual_rms': ..., ...}
```

## Послідовний цикл із зовнішньою моделлю

```python
from ensemble_da import Observations, run_cycle

def forecast(ensemble, t_from, t_to):          # ваша модель: просуває кожного члена ансамблю
    ...
    return ensemble_at_t_to

observations = {t: Observations(values, indices, R) for t, values in ...}
out = run_cycle(forecast, initial_ensemble, times, observations,
                mode='smoother', window=0.25, localization=Localization(...), rng=1)
out.analyses[t], out.forecasts[t], out.steps
```

Режими: `noda` (лише прогноз), `filter` (аналіз кожного моменту за його спостереженнями),
`smoother` (аналіз кожного моменту за спостереженнями в межах ± `window`). Аналізуються всі моменти,
крім останнього; кожен аналіз задає початковий стан прогнозу до наступного моменту, а згладжувач
прогнозує далі — до кінця наступного вікна, щоб майбутні стани вікна існували. `plan_cycle`
повертає план без виконання.

## Командний рядок

```bash
ensemble-da assimilate case.npz -o result [--transform log] [--no-perturb] [--seed 1]
```

`case.npz` містить `forecast`, `observations`, `observation_covariance`, `observation_indices` (або
`observation_operator`) і, за потреби, `localization`; аналіз записується в `result.npz`,
діагностика — в `result.json`.

## Відтворення експериментів

```bash
pytest -q                                      # тести, без зовнішніх даних
python examples/demo_transport_1d.py --plot demo.png
python benchmarks/verify_reference.py          # порівняння з незалежним розв'язувачем
python benchmarks/benchmark_analysis.py        # заміри часу; див. benchmarks/README.md
```

- **Twin-експеримент** ([examples/demo_transport_1d.py](examples/demo_transport_1d.py)): одновимірна
  модель адвекції–дифузії (200 комірок, 20 членів ансамблю, 50 логнормальних спостережень,
  невизначена швидкість) генерує прогнозний ансамбль і синтетичні спостереження відомого еталона.
  Виводить похибки прогнозу й аналізу для одного аналізу (з локалізацією й без) і для циклів
  noda/filter/smoother. Іграшкова модель лежить в `examples/`, поза пакетом.
- **Незалежний еталон** ([benchmarks/verify_reference.py](benchmarks/verify_reference.py)):
  робочий розв'язувач проти зважених найменших квадратів у просторі стану на 384 задачах, зокрема
  з виродженими ансамблевими коваріаціями.
- **Бенчмарки** ([benchmarks/benchmark_analysis.py](benchmarks/benchmark_analysis.py)): заміри
  етапів обох постановок із записом середовища; сирі результати в
  [benchmarks/results/](benchmarks/results/).

## Структура пакета

| Модуль | Вміст |
|---|---|
| `assimilate.py` | `assimilate`, `assimilate_window`, `AssimilationResult` — публічний етап аналізу |
| `analysis.py` | `analysis_update` — оновлення Калмана у просторі спостережень (Холецький) |
| `covariance.py` | `estimate_covariance`, `Localization`, `great_circle_distance` |
| `observations.py` | `observation_covariance`, `perturbed_observations`, `Observations` |
| `state.py` | `build_window_state` — скорочений багаточасовий стан вікна згладжувача |
| `transform.py` | захищені log / exp для логнормальних величин |
| `cycle.py` | `plan_cycle`, `run_cycle` — послідовний цикл із переданою функцією прогнозу |
| `reference.py` | `reference_analysis` — незалежний розв'язувач для верифікації та бенчмарків |
| `cli.py` | `ensemble-da assimilate` |

## Ліцензія

MIT — див. [LICENSE](LICENSE). Метадані для цитування: [CITATION.cff](CITATION.cff).
