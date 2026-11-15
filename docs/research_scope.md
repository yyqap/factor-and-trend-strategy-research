# Research scope and chronology

The projects began as internship prototypes. The original equity task used logistic regression, and the original allocation task implemented multi-asset trend and reversal rules. Subsequent independent research added model comparison, stricter time validation, portfolio and cost comparisons, risk attribution, historical industry diagnostics, and native-instrument accounting.

For Task 2 specifically, the internship contribution was a research-report-based Python rule implementation. The native-data backtest and its June 2020–August 2023 performance figures were completed in the November 2026 follow-up. Those dates describe the simulated investment window, not when the work was performed. Fixed rules and chronological execution alone do not establish an untouched out-of-sample evaluation.

The public package is a portable reimplementation of the documented research conventions. It is not a claim that every extension was completed during the original internship. The unchanged earlier scripts remain in `legacy/` so their outputs are not confused with the corrected work.

## What can be reproduced here

The synthetic demonstrations exercise data validation, causal signal generation, model fitting, selection, portfolio construction and accounting without the original input data. Their outputs are explicitly marked as synthetic and are not the historical results reported in `results/`.

The historical aggregate results record the existing evaluated runs. Reproducing their exact values requires the same original equity panel or the same native-contract marks, exchange calendars, FX conventions and declared configuration. Use the local-data interfaces and data dictionary rather than treating synthetic inputs or substitute instruments as exact reproductions.

## Distinct research questions

1. **Prediction:** do the equity model scores rank next-period stock returns? Monthly Rank IC and its time-series uncertainty address this question.
2. **Portfolio construction:** how do stock count, weights, turnover and state rules change hypothetical realized portfolio paths?
3. **Risk attribution:** how much return variation is associated with specified market and style factors? An intercept is conditional on that factor model.
4. **Industry robustness:** does the ranking association remain after controlling industry means using classifications already published at the signal date?
5. **Multi-asset implementation:** do trend signals, target exposures, trading calendars and funded-versus-futures accounting agree on when risk is taken and how P&L is earned?

Industry-adjusted IC is not an industry-neutral investment portfolio. A positive regression alpha is not a guarantee of future returns. A state overlay is a tested portfolio adjustment rather than proof of universally better risk control.
