> Archived script reference. Paths and methods below describe only the scripts in `legacy/`. Use the current [project README](../README.md) and [results guide](../docs/results_guide.md) for the validated project configuration and results.

# Archived strategy scripts

These archived scripts cover factor-based equity selection and multi-asset trend allocation. Raw datasets, source reports, reference PDFs and sensitive materials are excluded. Their methods and outputs are not interchangeable with the validated modules and aggregate results linked above.

## Task 1: Factor-Based A-Share Stock Selection

Task 1 develops a cross-sectional A-share stock selection model using a monthly factor panel. The model labels the top and bottom 30% of next-month excess returns, preprocesses 70 equity factors with MAD-based winsorization and standardization, and trains a logistic regression classifier to estimate stock-level positive-return probabilities.

The archived equity script then uses out-of-sample predictions to construct a monthly top-N portfolio weighted by predicted probability. The workflow covers factor preprocessing, supervised classification, out-of-sample forecasting, and basic portfolio backtesting.

Code: `src/task1_factor_stock_selection.py`

## Task 2: GMAT-Style Multi-Asset Trend Strategy

Task 2 implements GMAT-style global multi-asset allocation rules. The strategy covers equity indices, government bonds, and commodities, and combines trend-following signals with volatility-aware position sizing, asset-level caps, and periodic rebalancing.

The code includes GMAT2-style and GMAT3-style configurations. It uses reproducible synthetic price paths to validate the implementation structure, rather than publishing or relying on proprietary market data.

Code: `src/task2_gmat2_style_allocation.py` and `src/task2_gmat3_style_allocation.py`

## Repository Structure

```text
.
├── src/
│   ├── task1_factor_stock_selection.py
│   └── task2_gmat2_style_allocation.py
│   └── task2_gmat3_style_allocation.py
├── requirements.txt
└── README.md
```


## Notes

- Task 1 requires local factor CSV files, which are not included in this public repository.
- Task 2 uses simulated asset prices for implementation validation.
- This repository is for research demonstration only and does not provide investment advice.
