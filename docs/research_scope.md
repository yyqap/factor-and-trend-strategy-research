# Research design and implementation

The repository studies stock ranking and multi-asset allocation as complete research workflows. The equity pipeline combines factor preprocessing, linear and tree models, chronological validation, portfolio construction and risk diagnostics. The multi-asset pipeline implements research-report-based trend, recovery and volatility rules with native-instrument accounting.

The package exposes modular research components, explicit configurations and checked aggregate results. Archived implementation versions remain in `legacy/`; current methods and result definitions are documented in the task guides.

## Research questions

1. **Prediction:** do model scores rank next-period stock returns? Monthly Rank IC and its statistical uncertainty address this question.
2. **Portfolio construction:** how do stock count, weights, turnover and state rules change simulated portfolio paths?
3. **Risk attribution:** how much return variation is associated with market and style factors, and what intercept remains under the specified factor model?
4. **Industry robustness:** does ranking association remain after controlling industry means using classifications published by the signal date?
5. **Multi-asset implementation:** do signals, exposures, trading calendars and funded-versus-futures accounting agree on when risk is taken and how P&L is earned?

## Historical results and software demonstrations

The statistical figures and full backtest charts use the aggregate historical outputs in `results/`. Reproducing the historical model fits and constituent-level portfolios requires the same equity panel or native-contract marks, calendars, FX conventions and declared settings. Exact data windows, time splits and holding dates remain in the detailed methods, configurations and result files.

The optional synthetic demonstrations separately exercise data validation, fitting, selection and accounting with generated inputs. They do not produce the published historical figures or certify historical performance. The public aggregate CSVs are sufficient to regenerate the displayed statistical and backtest charts.

## Evaluation boundaries

Chronological fitting and causal execution do not make repeatedly examined history a fresh untouched holdout. Task 2 is a fixed-rule historical implementation, not a machine-learning train/test experiment or an exact replication of undisclosed index conventions.

Industry-adjusted IC is a ranking diagnostic, not an industry-neutral portfolio. Regression alpha is conditional on the factor model. Market-state overlays are evaluated risk adjustments, and their historical effects do not establish universally better risk control. Full returns, costs, drawdowns and unsuccessful comparisons remain available on the [results page](../results/README.md).
