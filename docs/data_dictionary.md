# Data dictionary

This dictionary describes the published **aggregate historical result files**. Public demo input schemas belong to the executable modules' documentation. Unless noted, rates are decimal fractions (`0.01 = 1%`), correlations and ratios are dimensionless, dates are ISO dates, and booleans are `True`/`False` in CSV. No row in these exports identifies an individual stock holding or prediction.

## Common Task 1 identifiers

- `model`: `ridge`, `logistic` or `lightgbm`.
- `config_id`: unique `model::catalog_group::policy_id`; the stable join key between the portfolio and attribution catalogs.
- `catalog_group`: `stock_book`, `market_overlay` or `relative_state`.
- `policy_id`: an actually evaluated rule, not an unrun combination.
- `conditional_stock_count`: number of stocks when the conditional stock basket is active.
- `candidate_pool`: model-ranked candidate count used by the rule. A defensive rule can choose 30 low-risk-ranked names from a larger model-ranked pool.
- `weighting`: `equal`, `risk_rank` or the preserved rule label.
- `regime_window_months`: historical lookback for the relative-state family; zero where not applicable.
- `cost_bps`: assumed cost per traded side, in basis points. One basis point is `0.0001` of traded notional.
- `selected_common_absolute_development`: selected under the common market/cash experiment's development MDD/return-retention criterion.
- `selected_holding_relative_development`: selected under the stock-count/weight experiment's development additive-relative-drawdown criterion.
- `selected_relative_state_development`: selected under the prior relative-state experiment's development criterion.
- `original_frozen_primary_model`: the model family selected by the shared development Rank-IC comparison. None of these flags selects the best later profit.

### Rule identifiers

- `top{30,50,100}_equal`: highest-scoring N distinct stocks, equal weights, fully invested.
- `top{30,50,100}_risk_rank`: same N, weights proportional to `1/(1+q)`, then normalized; `q=(average rank of the current full-universe processed std_3m factor−1)/(universe size−1)`. This is not inverse realized volatility.
- `trend_sma10`: Top30 conditional basket; stock exposure 1 if signal-end CSI300 close exceeds its ten-complete-month average, otherwise zero.
- `momentum12`: stock exposure 1 if twelve-month CSI300 price momentum is positive, otherwise zero.
- `vol63_target15`: stock exposure `min(1,0.15/sigma)`, where sigma is 63-day CSI300 realized volatility annualized with `sqrt(252)`. The 15% reference determines exposure; it is not achieved portfolio volatility.
- `trend_sma10_vol63_target15`: product of the trend indicator and volatility-scaled exposure.
- `always_defensive{100,200}`: full investment in 30 low-`std_3m`-ranked names within the model's stated candidate pool.
- `switch_w{6,12}_pool{100,200}`: use that defensive stock basket in a weak state, otherwise the original Top30. State uses the past six/twelve already-known equal-universe relative-return fields, through signal `t−2`. This rule holds stocks rather than cash.

The exact IDs present in the CSV are authoritative. The catalogs retain baseline aliases as one canonical `stock_book/top30_equal` configuration rather than counting the same design repeatedly.

## Task 1 portfolio performance

For chronological holding-period net returns `R`, same-period RF proxy `F`, benchmark returns `B`, and `A=R−B`:

- `months`: 49 holding periods.
- `first_entry_date`, `last_exit_date`: first and last valuation endpoints, July 1, 2014–August 1, 2018.
- `elapsed_days`, `elapsed_years_actual36525`: date difference and that difference divided by 365.25.
- `final_nav`: `product(1+R)`, starting from NAV 1.
- `total_return`: `final_nav−1`.
- `cagr`: `final_nav^(365.25/elapsed_days)−1`.
- `cagr_monthly_convention`: `final_nav^(12/months)−1`, a separate convention.
- `annual_arithmetic_return`: `12×mean(R)`; not CAGR.
- `annual_volatility`: `sqrt(12)×sample_std(R)`.
- `annual_arithmetic_excess_over_rf`: `12×mean(R−F)`.
- `sharpe`: `sqrt(12)×mean(R−F)/sample_std(R−F)`.
- `sortino`: `sqrt(12)×mean(R−F)/sqrt(mean(min(R−F,0)^2))`.
- `monthly_observation_nav_mdd`: maximum `1−NAV/running_max(NAV)`, including starting NAV 1. This is monthly observed **wealth** drawdown, not additive relative drawdown and not daily/intramonth MDD.
- `mdd_peak_date`, `mdd_trough_date`: observed dates of the peak and subsequent largest drawdown.
- `calmar`: CAGR divided by the monthly-observation NAV MDD.
- `tracking_error`: `sqrt(12)×sample_std(A)`.
- `information_ratio`: `sqrt(12)×mean(A)/sample_std(A)`; not Sharpe.
- `annual_arithmetic_active_return`: `12×mean(A)`; not relative-wealth CAGR.
- `positive_absolute_month_fraction`, `positive_active_month_fraction`: fraction with `R>0` and `A>0`; neither is stock prediction accuracy.
- `worst_holding_period_return`, `best_holding_period_return`: minimum and maximum net holding-period R.
- `mean_stock_exposure`: mean target stock exposure `g`; remaining weight is the research cash proxy for market/cash overlays.
- `cash_only_periods`: number of periods with `g=0`.
- `mean_two_sided_target_turnover`: average sum of absolute target stock-weight changes, counting buys and sells; initial entry included. It is not divided by two.
- `csi300_beta_ols`: OLS slope of strategy excess-over-RF returns on CSI300 excess-over-RF returns.
- `corr_with_csi300`: Pearson correlation between net strategy and CSI300 returns.
- `attribution_status`: states whether that fee row has attribution. The catalog estimates attribution at 10 bps only.

## Task 1 factor attribution

`factor_attribution.csv` has 48 rows, one per configuration. `attribution_cost_bps=10`, `hac_lag=6`, `months=49`. Fields beginning with `ff3_` and `ff5_` are parallel model results:

- `alpha_monthly`: OLS intercept in holding-period strategy excess-over-RF returns.
- `alpha_annualized_arithmetic`: `12×alpha_monthly`.
- `alpha_hac_t`: intercept divided by its HAC standard error.
- `alpha_p_normal`: two-sided normal-approximation p-value from that statistic.
- `alpha_ci95_low`, `alpha_ci95_high`: 95% HAC interval for **annual arithmetic** alpha.
- `r_squared`: explained return variation, not profit contribution.
- `loading_mkt_rf`, `loading_smb`, `loading_hml`: regression slopes on market excess, size and value factors.
- `loading_rmw`, `loading_cma`: FF5 profitability and investment slopes.

Factors are Chinese A-share BetaPlus daily series aggregated over `(entry_date,exit_date]`. Daily compounded long–short aggregation is a declared proxy; it is not assumed identical to publisher fixed-leg monthly factors. A loading is not a portfolio weight. Alpha is conditional on the model, dates and previously examined history; it is not a guarantee of stock-selection skill.

## Task 1 model/CV files

`model_comparison.csv` contains one row per family:

- `selected_by_development`: shared development-family winner.
- `cv_mean_rank_ic`: equal-weight mean of monthly full-universe Rank IC across 66 chronological validation months.
- `historical_full_universe_rank_ic`: mean of 49 later monthly full-universe Rank ICs.
- `industry_raw_matched_rank_ic`, `industry_adjusted_rank_ic`: same-sample primary industry comparison, described below.
- `selected_configuration`: JSON-encoded frozen within-family configuration; not a new test-period search.
- `baseline_cost_bps`, `cagr`, `sharpe`, `monthly_observation_nav_mdd`, `ff5_alpha_annualized_arithmetic`: join to the Top30 equal-weight baseline at 10 bps.

`cv_leaderboard.csv` has `model`, the prespecified `config` label and `mean_monthly_rank_ic`. It reports evaluated development configurations, not an estimate of a universal best model.

## Task 1 industry files

`industry_monthly_ic.csv` has 147 rows: three frozen model scores × 49 months, using minimum group size five. It contains no individual scores or industry members.

- `signal_month`: prediction decision month, before the evaluated holding return.
- `minimum_group_stocks`: minimum current eligible stock count per known industry, primary five.
- `total_original_stocks`, `included_stocks`, `industry_count`: original count, retained count and retained group count.
- `included_fraction`: retained / original count, after unknown and small-group exclusions.
- `full_original_rank_ic`: Spearman(score, future return) on the full original monthly universe.
- `raw_rank_ic_same_sample`: same correlation on exactly the retained stocks used by the adjustments.
- `industry_adjusted_rank_ic`: subtract each industry's mean from raw scores and raw future returns separately, then compute Spearman of the residuals.
- `industry_partial_spearman`: first average-tie-rank both original vectors; subtract industry means of ranks; calculate Pearson correlation of residual ranks without reranking.
- `return_only_group_adjusted_ic`: Spearman(original score, future return minus industry mean), the Alphalens-style one-sided definition.
- `within_industry_n_weighted_ic`: mean finite within-industry Spearman IC, weighted by group stock count.
- `industry_adjusted_minus_raw`: paired monthly difference between primary adjusted and raw same-sample IC.

`industry_inference.csv` includes minimum group sizes 2, 5 and 10. The `metric` field names one of the six monthly series above. `mean`, `months`, `hac_lag`, `hac_se`, `hac_t`, `hac_ci95_low`, `hac_ci95_high`, `positive_months` and `positive_month_fraction` describe inference across chronological monthly observations. HAC uses Bartlett weights, lag six, `N/(N−1)` correction and a normal 95% interval.

`industry_bootstrap.csv` contains the five-stock primary sample only. `block_length` is 6 primarily or 3/12 as sensitivities; `draws=10000`; `ci95_low`/`ci95_high` are percentile intervals of resampled monthly means. Paired comparisons use the same sampled month blocks.

`industry_coverage.csv` contains `signal_month`, `total_stocks`, `industry_matched_stocks`, `industry_unknown_stocks`, `industry_matched_fraction` and `matched_industries`. These are coverage counts **before** excluding small groups. Quarterly labels are the latest published by the decision date; UNKNOWN is not treated as one economic industry.

## Task 1 strategy-level time series

- `monthly_baseline_nav.csv`: `model`, `cost_bps`, `date`, `nav`, `drawdown`; 600 rows including starting NAV for each model/fee. `drawdown` is a positive observed loss from the running NAV peak.
- `monthly_baseline_returns.csv`: `model`, `cost_bps`, `signal_month`, `entry_date`, `exit_date`, `net_return`, `active_return`, `target_turnover_two_sided`. Returns are portfolio-level aggregates; the raw benchmark quote series and stock constituents are not distributed.

## Task 2 files

`performance.csv` contains the complete native window and a separately fixed January 2021–cutoff window:

- `window`: `full_native_window` or `fixed_2021_to_cutoff`.
- `initial_date`, `first_return_date`, `end`: initial NAV, first return and final valuation dates.
- `observations`: 836 daily returns for the full run, 685 for the fixed window.
- `years_elapsed`: actual elapsed calendar days / 365.25.
- `cumulative_return`: final NAV / starting NAV minus one.
- `cagr`: final/start NAV raised to `1/years_elapsed`, minus one.
- `annualized_volatility`: daily return sample standard deviation × `sqrt(252)`.
- `sharpe_rf_zero`: daily arithmetic mean / daily sample standard deviation × `sqrt(252)`, with RF assumed zero.
- `maximum_drawdown`: maximum daily observed `1−NAV/running_max(NAV)`, including initial NAV. It is not intraday drawdown.
- `drawdown_peak_date`, `drawdown_trough_date`: peak/trough observation dates.
- `positive_months`, `months_including_partial_endpoints`, `positive_month_fraction`: compounding daily returns within each calendar month, including the two partial sample endpoints.
- `daily_skewness`, `daily_excess_kurtosis`: daily distribution diagnostics from the completed run.

`strategy_nav.csv` has 837 rows: initial NAV 1 on June 3, 2020, then 836 daily observations. Columns are `date`, `is_initial`, `net_return`, `nav`, `drawdown`. Initial `net_return` is blank because no prior interval is claimed.

`monthly_returns.csv` has 39 rows with `month`, `first_return_date`, `last_return_date`, `observations`, `net_return`, `partial_sample_endpoint`. `yearly_returns.csv` analogously has `year` and `partial_year`; 2020 and 2023 are partial. The return is the product of actual reported net daily growth factors, minus one; partial periods are not annualized.

`universe.csv` contains the public model mapping: `logical_asset`, semicolon-separated `actual_legs` and corresponding `leg_weights`. There are 16 logical assets and 20 distinct underlying legs. This mapping is not a position or trade history.

`summary.json` records first execution, cutoff, fee/RF assumptions, accounting conventions and known limitations. `verification_summary.json` contains only aggregate reconciliation outcomes; it omits local paths, raw-source hashes and acquisition inventories.
