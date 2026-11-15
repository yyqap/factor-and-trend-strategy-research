# Data access and publication boundary

The public repository contains reusable research code, synthetic demonstrations, method documentation and **aggregate outputs** from completed historical studies. It does not include the underlying private factor panel, stock-level predictions or holdings, vendor quotes, exchange/vendor archive files, source PDFs, personal documents, acquisition manifests or local filesystem paths.

Access to this source code does not grant rights to third-party data. Downloadability is not permission to redistribute. Use data you are authorized to use and check the relevant provider's terms separately. No data-provider API credentials are embedded.

## What can be reproduced from this checkout

- The public demos and tests run using synthetic/generated fixtures rather than the historical source data.
- The five figures in `docs/figures/` can be regenerated with `python docs/figures/regenerate.py`, using public aggregate CSVs for statistical/backtest charts and the declared configuration for the workflow diagram. No underlying market input is read.
- Summary counts, catalog joins, strategy NAV compounding and many metric identities can be inspected directly from the aggregate exports.
- The complete historical model training, constituent-level portfolio construction and source-data verification require the omitted inputs. A successful synthetic demo must not be described as a reproduction of the published historical returns.

## Executable local interfaces

These commands match the installed public CLI. All example paths are repository-relative placeholders for caller-owned data; the files themselves are not supplied:

```bash
# A normalized equity panel or original numbered-month directory.
python -m factor_trend_research.task1 --input data/private/panel.csv --config configs/task1.json --output outputs/task1_local

# Add absolute reconstruction only after independently checking the holding horizon.
python -m factor_trend_research.task1 --input data/private/panel.csv --config configs/task1.json --benchmark data/private/matched_periods.csv --confirm-matched-horizon --output outputs/task1_absolute

# Prepared original-underlying multi-asset inputs, not arbitrary quote files.
python -m factor_trend_research.task2 --native-inputs data/private/task2 --config configs/task2.json --output outputs/task2_local
```

The Task 1 normalized CSV requires `stock,signal_month,target` plus numeric factor columns. `signal_month` is a monthly date such as `2014-06`; the target is already the next holding-period arithmetic relative return and must not be shifted again. A directory adapter accepts numbered `82.csv`–`243.csv`, with `stock,month,status,return` and numeric factors. It retains `status=1`; unobserved placeholder ID244 is excluded. A replacement dataset is not automatically the same historical universe.

An optional matched benchmark CSV requires `signal_month,entry_date,exit_date,benchmark_return,rf_return`, all return fields in decimals. `--confirm-matched-horizon` is an explicit caller assertion after source review, not software certification of vendor semantics. An optional `--industry-history` CSV requires `stock,industry,snapshot_date,published_date`; security keys must already match the panel.

Task 2's prepared directory requires `metadata.json` and exactly named input files:

- `accounting_intervals.csv`: `date,asset_id,start_price,end_price,start_fx,end_fx,held_contract,next_contract,next_price,tradeable`; 20 accounting legs for each interval after the initial normalization date. Funded indices have no contract fields; each futures return interval uses the same held contract at its endpoints.
- `local_actual_leg_returns.csv`, `cny_actual_leg_risk_returns.csv` and `native_observed.csv`: aligned date-indexed 20-leg matrices; the last contains explicit true/false observation masks.
- `common_execution_dates.csv`: `date,common_open`, with explicit true/false values consistent with the actual-leg trading masks.

Task 2 metadata must declare schema version 1, `input_kind:"native_original_underlyings"`, a reviewed `data_gate_passed:true`, evaluation dates, calendar/FX/roll policies, all 20 instrument specifications and SHA256 values for the five exact input files. Hashes lock reviewed local files; they do not establish ownership, source quality or point-in-time availability. See the complete [Task 1 input contract](task1_methodology.md#input-contract) and [Task 2 native input contract](task2_methodology.md#local-real-data-input-contract) before preparing data.

The configurations encode declared settings, not bundled source data. A local invocation writes its own output schema. The [code-to-result map](results_guide.md#how-the-code-configurations-and-archived-results-correspond) distinguishes direct CLI outputs, callable diagnostic helpers and archived reporting exports.

## Task 1 historical inputs

The historical study used a supplied monthly A-share panel with 70 fundamental and price/volume factors, stock identifiers, signal-month mappings, eligibility status and a forward stock-minus-CSI300 return field. No original panel rows are published. Historical release vintages for all supplied financial factors were not independently reconstructed; an apparently valid chronological model split does not certify the underlying factors as point-in-time.

A replacement dataset needs, at minimum:

- Stable security identifiers and one unique security/signal-month key.
- Factors available before the decision time, with disclosure/known-at dates for financial reports.
- An explicit eligibility and investability policy that preserves missing and delisted-name limitations.
- Precisely defined forward holding-period returns, benchmark identity and entry/exit dates.
- Enough chronological history for the stated training, validation and later evaluation windows.

Do not silently substitute another universe, current constituents, newer factor definitions, a total-return index for a price index, or calendar-month returns for the historical first-trading-close holding interval. Such changes create a new experiment and need their own labels.

### External risk and classification sources

The absolute-return reconstruction uses official **CSI300 price-index** history, available through [China Securities Index](https://www.csindex.com.cn/). Raw quotes are not redistributed here. The main factor attribution uses the [BetaPlus Chinese A-share FF3/FF5 daily dataset](https://www.factorwar.com/data/factor-models/). Its RF is a research rate proxy; exact holding-window aggregation and factor definitions are documented in the results guide. This repository publishes regression coefficients and uncertainty summaries, not the provider's factor time series.

Industry robustness used dated quarterly classification releases from the [CSRC industry-classification archive](https://www.csrc.gov.cn/csrc/c100103/common_list.shtml). The private research extraction preserved publication dates, snapshot dates and code/name checks. The latest release published by each signal-month end was used; later/current labels were not backfilled. These are CSRC detailed major industries, not the Shenwan industry standard. Only monthly coverage counts and aggregate IC diagnostics are published.

If reconstructing the industry diagnostic, retain the release/publication evidence and classification level. A quarter-end snapshot published in the next month was not known at quarter-end. Missing securities remain unknown; they are not assigned a future label or pooled into a synthetic industry. Public quarterly labels can still lag business changes.

## Task 2 historical inputs

The completed native-data study required separate valuation, signal-observation and execution calendars; five actual equity price indices; original delivery-specific futures prices; contract point values and roll mappings; and FX marks. Its sources included the relevant Chinese exchanges and quote vendors, CME and ICE risk/settlement archives, equity-index providers and Federal Reserve H.10 FX observations. Those underlying histories and archive files are not bundled.

Native identity matters. An ETF, another contract, a continuous series with undisclosed rolls or a generic US-dollar commodity index is not automatically a substitute for the stated underlying. The historical research retained explicit contract and quote-type checks. The public summary exports do not supply enough information to recreate that source audit.

The 16 logical assets map to 20 underlying legs in `results/task2/universe.csv`. Five equity series are modeled as funded price-index exposures excluding dividends. Futures contribute held-contract variation margin rather than notional principal to NAV. The research's roll rules, ferrous-basket weighting and synchronous end-of-day sizing are declared implementation assumptions; they are not a claim to have recovered undisclosed reference-index rules.

The source search starts in 2018, but the complete strategy **does not** start earning returns in 2018. Its first eligible execution is June 4, 2020 after native-observation warm-up and the required monthly execution phase. Unknown active-source gaps were not converted into zero returns to force an earlier start. The published evaluation ends August 18, 2023.

The FX convention uses the previous observed Federal Reserve H.10 mark. The historical release timestamps and revisions were not reconstructed. Carrying a prior observed date is therefore an observational convention, not a certification that the feed was publicly available at the exact historical decision time.

## Reference material and performance claims

The private reference report and its annual return quotations are not redistributed. No exact daily reference NAV was available for an equal-window Sharpe, volatility or MDD comparison, and the public exports explicitly flag this absence. The repository reports its complete native-run results using the stated account and metric conventions.

All performance is hypothetical historical research under the stated fees and accounting assumptions. The source inputs and historical evaluation period have been examined repeatedly; the exports are not an untouched out-of-sample investment record. No inferred daily Task 1 NAV or unrun combined strategy is introduced to improve the presentation.
