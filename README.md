# Polymarket Forecasting & Market Efficiency

A research pipeline that evaluates prediction-market prices as probability
forecasts and tests whether recalibration or machine learning improves them.
It combines daily YES-token price histories from Polymarket's CLOB API with
market metadata from the Gamma API. The data are daily prices and metadata;
historical order-book depth and order flow are not used.

Two linked studies cover calibration, favorite–longshot bias, and transaction
costs, followed by a monthly walk-forward comparison of logistic recalibration,
isotonic regression, and LightGBM models with price, path, structure, and text
features.

## Research questions

1. Are Polymarket prices calibrated probability forecasts, and how does
   calibration change as the forecast horizon lengthens?
2. Do prices carry systematic biases, such as a favorite–longshot bias or a
   premium on the YES side?
3. Can simple strategies that trade against those biases make money after a
   1¢ per-contract trading cost?
4. Can recalibration or a machine-learning model beat the market price out of
   sample, and does trading on model–price disagreement survive costs?

## Pipeline

```text
01_fetch_markets   Gamma API market metadata
       |
       v
03_build_sample    binary, closed CLOB markets, volume >= $1,000
       |
       v
02_fetch_prices    CLOB API daily YES prices for the sampled markets
       |
       +--------------------------------+
       v                                v
Calibration study                    Forecast-model study
04_build_panel  market x horizon     08_features    snapshots for both anchorings,
                panel, 1-90 days,                   price-path and structure features
                two date anchorings  09_model       category and TF-IDF/SVD text features,
05_analysis     calibration, bias                   monthly walk-forward model ladder
                tests, backtests     10_interpret   diagnostics, divergence backtest
06_figures      figures/calibration/ 12_supplement  blends, mature folds
07_tables       markdown tables      11_figures2    figures/models/, tables
```

Each numbered script is one stage that reads saved outputs from earlier
stages, so a stage can be rerun on its own once its inputs exist.

## Results

The included results were regenerated on **18 October 2026**. Relative to the
first saved estimates, the pipeline now enforces forecast timing and event-size
availability, pools tied prices in the calibration decomposition, tunes blends
only on earlier data, places prices that sit exactly on a bin edge in the bin
that edge opens, and gives the model stages a total row order, so each
walk-forward fold depends only on data available before its test month.

Sample construction starts from 72,578 closed-market metadata records and
applies its filters in sequence: 44,388 are binary YES/NO markets;
40,484 of those have terminal 0/1 outcome prices and a resolution status
that is `resolved` or not recorded; 40,476 also have two distinct CLOB token
IDs, an order book that is not disabled and a closure-time proxy;
34,792 have a scheduled end date on or before 31 July 2025;
30,680 have lifetime volume of at least $1,000; and 26,967 were
listed for at least 1.5 days before closure. Explicitly unresolved statuses,
including proposed and disputed, are excluded. The
[construction counts](results/sample_construction.json) store the starting
number under the legacy key `all_resolved_markets`.

Of the 26,967 sampled markets, 26,508 have usable price histories and
26,499 contribute 82,610 calibration observations across seven horizons from
1 to 90 days. The model feature dataset has 135,522 rows across both date
anchorings.

### Calibration

A calibrated price has a log-odds slope of 1 and an intercept of 0. Each row
uses the latest valid quote a fixed number of days before the closure-time
proxy. Standard errors are clustered by event.

| Horizon (days) | Observations | Events | Brier score | Skill vs. base rate | Miscalibration | Log-odds slope (SE) | Log-odds intercept (SE) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 26,485 | 7,702 | 0.0836 | 0.508 | 0.0011 | 1.106 (0.020) | -0.068 (0.027) |
| 3 | 20,952 | 6,473 | 0.0849 | 0.481 | 0.0021 | 1.052 (0.023) | -0.117 (0.050) |
| 7 | 13,439 | 4,509 | 0.0748 | 0.519 | 0.0011 | 1.075 (0.025) | -0.050 (0.037) |
| 14 | 9,420 | 3,345 | 0.0624 | 0.566 | 0.0013 | 1.128 (0.030) | -0.102 (0.045) |
| 30 | 6,256 | 2,279 | 0.0657 | 0.530 | 0.0029 | 1.120 (0.033) | -0.258 (0.049) |
| 60 | 3,595 | 1,268 | 0.0699 | 0.502 | 0.0045 | 1.088 (0.042) | -0.376 (0.071) |
| 90 | 2,463 | 840 | 0.0727 | 0.433 | 0.0080 | 1.006 (0.047) | -0.619 (0.078) |

Prices are informative at every horizon: the Brier score is 43% to
57% below that of a constant forecast at the horizon's base rate.
The slope exceeds 1 by more than two standard errors from 1 to 60 days and is
1.01 at 90 days. The intercept lies between -0.05 and -0.12 from 1 to 14
days and falls to -0.62 at 90 days. Miscalibration is the CORP decomposition's calibration term, which
pools tied prices before isotonic fitting; its identity is
`Brier = miscalibration - discrimination + uncertainty`.

![Calibration by forecast horizon](figures/calibration/fig1_calibration_by_horizon.png)

Calibration-error intervals resample entire events and recompute both mean
outcomes and mean prices. Closure dates are proxies, as explained below.

### Price biases and bias strategies

A slope above 1 is the favorite–longshot pattern: low-priced contracts resolve
YES less often than their prices imply. Mean outcome minus price by price
bucket, in percentage points with event-clustered t in parentheses:

| Closure-proxy horizon | Price below 10% | Price 10–90% | Price 90% or above |
| --- | ---: | ---: | ---: |
| 7 days | -0.60 (-4.56) | -1.47 (-2.39) | +0.90 (1.70) |
| 30 days | -1.20 (-6.61) | -5.37 (-6.70) | +1.59 (1.96) |
| 90 days | -1.43 (-4.60) | -10.50 (-7.87) | -1.30 (-0.63) |

YES is overpriced below 90% at each of these horizons, and the mid-range gap
widens from 1.47 points at 7 days to 10.50 at 90. The log-odds intercept is
negative at all seven horizons, from -0.050 (SE 0.037) at 7 days, which is not
distinguishable from zero, to -0.619 (SE 0.078) at 90 days.

![Gross return to buying YES by price bin at 7 days](figures/calibration/fig4_decile_returns.png)

Strategies that trade against these biases, per trade and held to settlement
(design under [Backtest design and limits](#backtest-design-and-limits)):

| Strategy | Trades | Gross mean return | After 1 cent | Event t | Monthly t |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 days, buy YES at 90–99% | 626 | 1.33% | 0.28% | 0.42 | 0.25 |
| 7 days, buy NO when YES is 1–10% | 4,056 | 1.15% | 0.11% | 0.43 | 1.16 |
| 30 days, buy YES at 90–99% | 328 | 1.71% | 0.65% | 0.72 | -0.33 |
| 30 days, buy NO when YES is 1–10% | 2,326 | 1.98% | 0.93% | 3.04 | 1.66 |
| 30 days scheduled-end, buy YES at 90–99% | 277 | 1.38% | 0.33% | 0.31 | -0.77 |
| 30 days scheduled-end, buy NO when YES is 1–10% | 1,969 | 1.69% | 0.63% | 1.74 | -0.34 |

Both t statistics refer to the return after 1 cent. Only the 30-day
closure-proxy NO strategy keeps an event t above 2 after costs (3.04), and it
weakens to 1.74 when quotes are anchored to scheduled end dates. Weighting months
equally, no strategy's monthly t exceeds 1.66. The biases are statistically
clear but are not reliably tradable after a 1-cent cost.

### Forecast models

The primary scheduled-end comparison has **55,826 evaluated observations,
25 test months and 6,021 event clusters**. These are
market–horizon observations, not distinct markets. Lower Brier scores are better.

| Forecast | Pooled Brier score |
| --- | ---: |
| Market price | 0.0785305 |
| Isotonic recalibration | 0.0777421 |
| Logistic recalibration | 0.0778376 |
| Full LightGBM | 0.0798941 |

Isotonic has the lowest pooled score in this comparison, a 1.00% reduction
relative to the price. Its event-clustered t is 2.42, while the equal-month
t is -0.32. Logistic reduces pooled error by 0.88% (event t = 4.46;
monthly t = 1.53). The full GBM is 1.74% worse than the price in this
comparison (event t = -2.59; monthly t = -3.73). These differences do not
establish consistent prospective forecast gains.

The split-sample GBM blend chooses weight 0.10 using 5,501 early
predictions whose closure-time proxies precede the 2024-07 cutoff, then
evaluates 49,405 later predictions from disjoint events. It lowers the Brier
score by 0.000307 (event t = 4.91; monthly t = 3.87), about
half of the 0.000637 that logistic recalibration alone gains
on the same later predictions. The full-sample blend curves are descriptive,
not tuning evidence.

![Model comparison](figures/models/fig2_1_model_ladder.png)

Error bars above are 1.96 times event-clustered standard errors. Equal-month
inference answers a different question and is reported separately.

In the closure-proxy robustness comparison (70,673 evaluated observations,
27 test months and 7,039 event clusters), the price scores 0.0744532.
Isotonic scores 0.0742677 (event t = 1.42; monthly t = -0.43),
logistic 0.0743569 (1.46; -1.22) and the full GBM
0.0740909 (0.95; 0.73). The GBM without text features has the
lowest pooled score, 0.0730686, a 1.86% reduction relative to the price
(event t = 3.82; monthly t = 1.28). Model rankings therefore depend on the date
anchoring. This comparison is retrospective: its snapshots and horizon feature
are measured back from a closure time that is not known in advance.

### Backtest design and limits

Calibration backtests buy YES at prices 90–99% or buy NO when YES costs 1–10%,
at 7- and 30-day closure-proxy horizons, plus a 30-day scheduled-end comparison.
The model backtest buys the side favored by the full GBM when its forecast
differs from the price by more than 2, 5 or 10 percentage points.

Each observation receives an equal monetary stake. Entry cost is price plus
1 cent per contract; the position is held to settlement. Buying NO costs
`1 - p + 0.01`. Return is payoff divided by entry cost minus one. A losing
trade loses the full stake; some winning outcomes merely break even after costs.
There is no compounding, capital budget, position cap or executable bid/ask model.
Repeated horizons and related markets can represent overlapping positions.

| Model disagreement threshold | Trades | Mean return after 1 cent | Event t | Monthly t |
| --- | ---: | ---: | ---: | ---: |
| 2 points | 24,694 | -5.39% | -3.35 | -1.97 |
| 5 points | 13,391 | 1.89% | 0.93 | 0.29 |
| 10 points | 6,234 | 7.98% | 2.36 | 0.94 |

These are price-based signal tests, not evidence of achievable portfolio returns.
Calibration backtests group monthly returns by closure-proxy month; model
backtests and forecast comparisons group by snapshot month. Cluster mean tests
use asymptotic CR0 errors. Monthly t statistics assume independent month means;
they are not adjusted for serial correlation. Model selection and multiple
comparisons on the same retrospective dataset remain limitations.

See the [calibration aggregates](results/analysis.json),
[model metrics and fold diagnostics](results/models/model_metrics.json),
[interpretation and backtests](results/models/interpretation.json), and
[supplementary comparisons](results/models/supplement.json).

## Run offline

Use Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python sample/run_sample.py
```

On Windows activate with `.venv\Scripts\Activate.ps1`. LightGBM may need
`libomp` on macOS. The sample runner creates a separate temporary project, runs
`00`, `03`, `04`, `05`, `06`, `07` and `08`, and reports its output path.
Use `--workdir DIR` to choose a new or empty directory. It preserves the included
results and generates all six calibration charts, even when long horizons have
too few observations. The bundled runner demonstrates early stages; its small
synthetic inputs are for software checks, not empirical research.

The tests cover malformed input, retries and interrupted resume, timing and
event availability, single-class fitting, tied-price calibration, uncertainty,
blend tuning, sample safety, and reconciliation of saved aggregates. They do not
recreate the historical model fits. Every script uses paths relative to its own
location, so relocated copies and execution from another directory are supported.

## Data and full reproduction

Only synthetic raw inputs, aggregate results and figures are included here.
Historical raw records, processed datasets and individual predictions are not
bundled. This folder alone cannot reproduce the historical estimates.
The synthetic sample has 293 invented metadata rows and 231 history records,
including intentionally failed, empty, single-print and gapped histories.
`python sample/make_sample.py` regenerates it deterministically.

For a historical rerun, supply `data/raw/markets_meta.jsonl` and
`data/raw/price_histories.jsonl` in a separate project copy, then run:

```bash
python code/00_setup.py
python code/03_build_sample.py
python code/04_build_panel.py
python code/05_analysis.py
python code/06_figures.py
python code/07_tables.py
python code/08_features.py
python code/09_model.py
python code/10_interpret.py
python code/12_supplement.py
python code/11_figures2.py
```

`12` must precede `11`, which reads its supplementary comparisons. Each stage
overwrites its generated outputs. The complete sequence was last executed on
18 October 2026 with Python 3.11.7 and the pinned dependencies.
Random seeds are 42; numeric results can vary between library builds and platforms.
For efficient fitting, set `OPENBLAS_NUM_THREADS=1` and
`VECLIB_MAXIMUM_THREADS=1` before starting Python; LightGBM itself uses four threads.

For new collection, start with an empty `data/raw/` in another copy, run `00`,
then `01_fetch_markets.py`, `03_build_sample.py`, `02_fetch_prices.py`, and
continue from `04` above. Metadata collection requires `curl`. The price request
uses a token ID, `interval=max` and `fidelity=1440`, as described in the
[history API reference](https://docs.polymarket.com/api-reference/markets/get-prices-history).
The [market API reference](https://docs.polymarket.com/api-reference/markets/list-markets)
documents metadata and date filters.

The collectors validate responses, retry transient failures with a finite limit,
checkpoint completed metadata windows atomically, and retry previously failed
price requests on resume. Price failures exit with an error after preserving
completed work. Successful empty histories are skipped on resume. Interrupted
final JSONL fragments are preserved separately before append recovery; malformed
complete records and conflicting timestamps fail explicitly. Metadata pagination
splits crowded date windows and reports a failure if the window cannot be split
further. Successful saved IDs are retained rather than refreshed.

Bounded live smoke checks retrieved one Gamma market and one CLOB history on
1 October 2026. A complete new collection was not run. API coverage, pagination
behavior and metadata can change, so these checks do not prove universe completeness.

## Design and limits

- Forecast snapshots must be at or after market creation and strictly before
  `t_res`, the closure-time proxy (`closedTime`, with `endDate` fallback).
  This proxy does not independently establish outcome availability.
  Missing creation dates use the first observed quote as the snapshot lower
  bound; the metadata lifetime filter cannot exclude those unknown lifetimes.
- Labels require a closed market and terminal outcome prices of 0 and 1.
  An explicit resolution status must be `resolved`. Legacy missing statuses
  remain eligible under the closed/terminal-price rule; those labels and the
  metadata were not independently verified against historical oracle records.
  [Proposal and dispute are separate stages from final resolution](https://docs.polymarket.com/concepts/resolution).
- Prices and path features use only quotes at or before the snapshot, with a
  1.5-day staleness limit. Precreation and invalid quotes are excluded. Probabilities
  are clipped to `[0.001, 0.999]` for modelling and log-odds calculations.
- Event size counts sampled markets created by the snapshot; first observed
  quotes supply a fallback when creation is missing. Future-created siblings
  do not contribute. Lifetime volume and current liquidity are excluded as
  predictors, although lifetime volume still determines sample selection.
  Invalid scheduled lifetimes become missing features; positive lifetimes use
  the actual denominator, even for unusually short schedules or late closures.
- Monthly training requires both snapshot and closure proxy to precede the
  test month. Test events appearing in training are excluded. The first six
  observed snapshot months form burn-in; folds need at least 500 training rows.
- TF-IDF/SVD is fitted only on eligible burn-in data. Early stopping holds out
  entire events ordered by their latest training snapshot. Its selected tree
  count is then refitted on every eligible training row. Single-class cases
  use explicit binary probabilities or the fixed training budget.
- Blend tuning uses only closure proxies preceding its fixed cutoff, followed by
  evaluation on later rows from different events. Fold sizes and timestamp
  bounds are recorded in the model metrics; predictions retain snapshot times.
- Questions, scheduled dates, event membership and the resolved-market universe
  were collected retrospectively. These corrections remove identified timing
  violations but do not establish a fully point-in-time dataset. Closure-proxy
  anchoring is retrospective; scheduled-end anchoring also relies on historical
  metadata accuracy. This work does not establish prospective profitability.

## Files

```text
code/                 Collection, construction, analysis, models and charts
sample/               Invented inputs, generator and isolated runner
tests/                Unit regressions, sample workflow and aggregate checks
results/              Calibration aggregates and bins
results/models/       Model metrics, interpretation and supplementary results
figures/calibration/  Six calibration figures
figures/models/       Six model figures
requirements.txt      Python 3.11 dependency pins
LICENSE               MIT license for original code and documentation
```

## License

Original code and documentation use the [MIT License](LICENSE).
Third-party data remain subject to their respective terms.
