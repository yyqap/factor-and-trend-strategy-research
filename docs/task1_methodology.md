# Task 1: factor-based A-share stock-selection research

This module reconstructs the research as a portable, explicit pipeline. The
original internship prototype used logistic regression. The linear/tree model
comparison, stricter validation, alternative portfolios, industry checks and
risk attribution are subsequent research. Synthetic demo results are software
fixtures and must never be described as historical investment performance.

## Run it

From the installed project:

```bash
python -m factor_trend_research.task1 --demo --output outputs/task1_demo
python -m factor_trend_research.task1 --demo --models ridge,logistic,lightgbm --output outputs/task1_demo_all
python -m factor_trend_research.task1 --input /path/to/local/csv_demo --config configs/task1.json --output outputs/task1_local
```

The demo runs Ridge and Logistic unless models are explicitly specified.
LightGBM is an optional dependency; requesting it without the library/runtime
raises an explanatory error. It is never replaced by another tree algorithm.
The complete local research configuration requests all three families.
On macOS, a LightGBM installation may additionally require the system OpenMP
runtime, available with `brew install libomp`. See the official
[LightGBM installation guide](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html#macos).
No developer-specific library path is required or embedded in this project.

### Executable scope versus archived results

The main CLI performs the three-family development selection, annual later
refits, and the **six stock-count/weight books at four fee assumptions**. It
does not automatically run all 48 archived portfolio configurations. Market
overlays and relative-state policies are available as explicit helper APIs;
the two always-defensive controls use `always_defensive=True` in
`relative_defensive_holdings`. Inputs for market, RF, factor returns and dated
industry membership are not bundled.

The archived `selected_*_development` portfolio flags came from a separate
causal development simulation with nested, then-available model tuning, over
2008-11–2014-04. The baseline CLI does **not** recreate that nested development
simulation or reselect portfolio/regime rules. In particular, its final
2014-selected parameters must not be backcast to generate earlier development
scores. The archived flags are reported historical experiment outcomes, not
outputs inferred from the public demo or a test-period portfolio sort.

Portable stock-book outputs correspond to `catalog_group=stock_book` in the
archive. Output `cagr_actual_days` maps to archived `cagr`,
`monthly_boundary_nav_mdd` to `monthly_observation_nav_mdd`, and
`additive_relative_drawdown_percentage_points` to the former relative-field
pullback diagnostic. The executable does not recreate source acquisition,
provider provenance or every auxiliary sensitivity result from aggregate CSVs.
Matching historical inputs and software versions are needed for numerical
parity; the package does not promise bitwise equality across library versions.

## Input contract

Two adapters are available:

- A normalized CSV with `stock,signal_month,target` and numeric factor columns.
- The original numbered monthly layout `82.csv` through `243.csv`, either in
  the supplied directory or its `csv_demo` subdirectory. ID 1 means 1998-04;
  the adapter retains `status=1` and maps `return ` directly to `target`.

In the original layout, the 162 signal months run 2005-01 through 2018-06.
ID 244 contains unobserved zero-placeholder labels and is excluded. The 70
factors span fundamental valuation/profitability/growth, price/volume,
volatility, turnover and analyst characteristics. The adapter does not assert
that arbitrary replacement data has the same number of factors or securities.

The target must already be the **next holding-period arithmetic stock return
minus the matching CSI300 return**, in decimal units. Do not shift it again.
The original data's timing was reconciled to first-trading-close of month `t+1`
through first-trading-close of `t+2`; it is not an ordinary month-end return.
Duplicate stock-month keys and missing/unrealized targets are rejected.
Raw source files, individual real predictions and holdings are not distributed.

Publication timestamps, revisions, delisted-stock coverage, trading eligibility
and upstream transformations still require the owner's vendor/data review.
`status=1` is supplied eligibility, not a new certification of tradability.

## Cleaning and feature selection

All preprocessing is refitted independently in each historical training window.
Nonfinite factor values become missing; training medians fill them, followed by
training-only mean/standard-deviation scaling:

`z=(x-mean_train)/std_train`.

Preprocessing uses every eligible training row, including the classifier's
middle 40%. All-missing or constant training columns are excluded. Validation
and future observations cannot change the imputer, scaler or factor filter.

For each training month, calculate factor-pair Spearman correlations. Summarize
each pair by the time median of its absolute correlation. The greedy filter
uses threshold 0.90, prioritizing factors by absolute mean training monthly
Rank IC and retaining one representative among highly correlated candidates.
This threshold and common filter are research choices, not universal optima;
tree interactions can make correlated features useful. Feature-screening logs
record retained factors and their representatives.

The supplied factors already appeared processed; the main pipeline does not
add another MAD clipping layer. `cross_sectional_mad_clip` is a separate optional
utility for genuinely raw inputs: clip each monthly factor at
`median ± k*median(abs(x-median))`; zero-MAD columns remain unchanged. Its MAD is
unscaled. It is not silently enabled for the historical project.

## Chronology and availability

The 113-month nominal development interval is 2005-01–2014-05; the 49-month
subsequent interval is 2014-06–2018-06. Same-month stocks always stay together.

The three expanding validation folds are:

1. Train 2005-01–2008-09; validate 2008-11–2010-08.
2. Train 2005-01–2010-07; validate 2010-09–2012-06.
3. Train 2005-01–2012-05; validate 2012-07–2014-04.

Prediction signal month `t` uses training signal months no later than `t-2`.
The gap prevents unavailable target outcomes from entering fitting. Each fold
fits once for its 22-month validation block. Final evaluation instead refits
annually using an expanding historical window; selected hyperparameters remain
fixed. The difference between these schedules is a limitation, not concealed
as identical validation. This already-examined historical interval is not a
fresh untouched holdout, and chronological splitting cannot undo prior research.

## Models and selection

- **Ridge:** continuous relative-return target; six lambda values from 0.0001
  through 10. Its objective is mean squared error plus `lambda*||b||²`, so
  sklearn's SSE-based `alpha=n_train*lambda`. The intercept is unpenalized.
- **L2 Logistic:** each month, exactly `floor(0.30*N)` weakest/strongest names
  receive 0/1; the middle is omitted from classifier fitting. Ties use stock
  code. Compare `C=0.01,0.1,1,10`; LBFGS, 1000 maximum iterations, tolerance 1e-6.
- **LightGBM:** the same two-tail classification labels; leaves 7/15,
  minimum leaf samples 200/1000, trees 100/300; learning rate 0.05,
  L2 leaf penalty 1 and fixed seed 20261109. No test-period early stopping.

Every validation month is scored across the entire eligible universe. Choose
each family's parameters, then the family, using the average of 66 monthly
full-universe Spearman(score,target) values. The class 1 output is used as a
ranking score; tail-only training does not calibrate the probability of a
full-universe stock finishing in the top 30%. Full-universe IC selection also
does not guarantee the best concentrated Top30 portfolio.

## Portfolio construction and costs

Rank by score descending and stock code ascending. Select exactly 30, 50 or 100
different names. Equal weights are `1/N`. The risk-rank variant computes
`q=(average_rank(std_3m)-1)/(universe_size-1)` across the **entire** current
universe, then normalizes `1/(1+q)` among selected names. This bounds the
unnormalized weight ratio by 2. The processed`std_3m` field is a characteristic,
not a certified annualized volatility estimate for direct inverse-vol weighting.

Costs are `bps/10000 * sum(abs(w_new-w_old))`, counting purchases and sales.
First entry is charged; no extra terminal liquidation is assumed. Compare
0, 5, 10 and 20 bps without choosing a different configuration from test results.
These are target-weight cost proxies: holding drift, impact, spread, taxes,
capacity, suspensions and price-limit constraints are not simulated.

## Relative versus absolute results

Without a matched benchmark/RF input, report relative arithmetic mean, tracking
error, information ratio and **additive relative pullback in percentage points**.
Do not compound the arithmetic relative field into account NAV or call its
information ratio a Sharpe ratio.

Optional benchmark CSV columns are
`signal_month,entry_date,exit_date,benchmark_return,rf_return `. Dates and decimal
returns must match the exact target horizon. The CLI requires
`--confirm-matched-horizon` after independently checking this. Date checks do
not certify vendor semantics or execution prices.

```bash
python -m factor_trend_research.task1 --input panel.csv --benchmark matched_periods.csv --confirm-matched-horizon --output outputs/task1_absolute
```

For stock exposure `g` and conditional-stock relative return `a`:
`Rp=g*(a+Rb)+(1-g)*Rf-cost`. NAV starts at 1 and compounds `1+Rp`.
CAGR uses actual elapsed days/365.25. Sharpe is
`sqrt(12)*mean(Rp-Rf)/sample_std(Rp-Rf)`. IR substitutes `Rp-Rb`.
MDD is `max(1-NAV/running_max_NAV)`, including the initial 1. This is a
holding-boundary/monthly-observation MDD; monthly data cannot recover daily or
intraday drawdown. Absolute reconstruction is conditional on valid matched inputs.

## Optional evaluation and regime helpers

`diagnostics.asof_industries` uses only the latest published snapshot available
at signal month-end, retaining unknown names. `industry_rank_ic` excludes
unknowns and groups smaller than the prespecified minimum 5. It compares the
raw IC on the identical sample with three distinct calculations:

- Spearman after demeaning raw scores and returns by industry.
- Partial Spearman: globally rank first, demean rank vectors, then Pearson.
- Alphalens-style IC: demean future returns only, leaving scores unchanged.

These are ex-post evaluation controls, not new predictions or neutralized
portfolio returns. The raw-score method depends on score scale; probability
and log-odds transformations need not produce the same demeaned ranking.
Quarterly classifications can lag business changes. A later/current industry
snapshot must never be backfilled into earlier decisions.

The optional industry CLI writes all minimum-group sizes 5, 2 and 10, using
the same metric names as the archived industry files. It also writes HAC6
monthly-mean inference and the primary five-stock sample's 10,000-draw
noncircular moving-block intervals, with block lengths 6 primarily and 3/12
as sensitivities, seed20261114. Within-industry ICs are additionally averaged
using valid group stock counts. An undefined monthly IC raises an error rather
than silently becoming zero.

`diagnostics.factor_attribution` accepts matched holding-date factor returns,
regresses `Rp-Rf` on an intercept and the chosen traded-return factors, and
computes Bartlett HAC inference. Alpha is 12 times the monthly intercept, not
the mean OLS residual or CAGR. A market-excess factor is not reduced by RF a
second time. Source-defined factor construction and daily-to-holding-period
aggregation remain explicit input assumptions.

`regimes.market_exposure ` implements signal-month-end benchmark SMA10,
12-month momentum, 63-day realized-volatility scaling `min(1,0.15/vol)` and their
trend×volatility combination. Only history through the signal month is used;
remaining capital earns the matched RF proxy. The15%reference is for benchmark
volatility scaling, not a guarantee of portfolio volatility.

`regimes.relative_defensive_holdings` is the earlier, different extension:
sum the last 6 or 12 known universe-average relative targets, using only signal
months`<=t-2`; when negative, pick the 30 lowest-risk names from the score
Top100 or 200. Equal-risk ties prefer higher model score, then stock code.
It stays fully invested and changes the stock basket. Its output
can be passed as`selected_holdings` to`backtest`. Neither helper picks a rule
from subsequent performance; a new selection experiment needs its own frozen
development protocol. Improving one historical drawdown is not proof of a
reliable future risk-control rule.

## Outputs and limits

The CLI writes selected configurations, the complete CV leaderboard, per-fit
feature logs, annual fit cutoffs, local predictions, all configured portfolio
sizes/weights/fees, and a data/limitations manifest. Keep real-data outputs
private; the repository's demo results, if included, are explicitly synthetic.

Positive IC, factor-regression alpha and a profitable constrained portfolio
answer different questions. Industry adjustment alone does not remove size or
other within-industry risks. Reliable alpha still requires genuinely new
periods, source timing checks, selection-aware inference and realistic trading
constraints. This module is research infrastructure, not a live execution system.
