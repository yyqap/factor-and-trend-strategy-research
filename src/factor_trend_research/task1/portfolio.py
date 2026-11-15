"""Unique-name portfolios and explicit relative versus absolute accounting."""
import numpy as np
import pandas as pd


def target_turnover(current, previous):
    """Two-sided L1 turnover; first investment is charged, final exit is not added."""
    return float(sum(abs(current.get(s, 0.)-previous.get(s, 0.))
                     for s in set(current) | set(previous)))


def select_holdings(group, n=30, weighting="equal", risk_feature="std_3m", exposure=1.):
    if group.stock.duplicated().any():
        raise ValueError("A monthly selection must contain unique securities")
    if len(group) < n or n < 1 or not 0 <= exposure <= 1:
        raise ValueError("Invalid number of holdings or stock exposure")
    if not np.isfinite(group[["score", "target"]]).all().all():
        raise ValueError("Scores and matched forward returns must be finite")
    chosen = group.sort_values(["score", "stock"], ascending=[False, True], kind="stable").head(n).copy()
    if weighting == "equal":
        raw = np.ones(n)
    elif weighting == "risk_rank":
        if risk_feature not in group or not np.isfinite(group[risk_feature]).all():
            raise ValueError("Risk-rank weights require a finite risk characteristic for every stock")
        ranks = group[risk_feature].rank(method="average")
        q = (ranks-1)/(len(group)-1) if len(group) > 1 else ranks*0
        raw = 1/(1+q.loc[chosen.index].to_numpy())
    else:
        raise ValueError("Weighting must be equal or risk_rank")
    chosen["weight"] = exposure*raw/raw.sum()
    return chosen


def validate_benchmark(benchmark):
    """Dates must describe the SAME first-close-to-first-close target horizon.

    This validates alignment structure, not the provenance of vendor returns.
    The caller must independently verify the target's arithmetic definition.
    """
    required = {"signal_month", "entry_date", "exit_date", "benchmark_return", "rf_return"}
    if not required.issubset(benchmark) or benchmark.signal_month.duplicated().any():
        raise ValueError("Matched benchmark requires unique signal_month, entry/exit dates, benchmark_return and rf_return")
    result = benchmark.copy()
    for col in ["entry_date", "exit_date"]:
        result[col] = pd.to_datetime(result[col])
    for row in result.itertuples():
        month = pd.Period(row.signal_month, "M")
        if not (row.entry_date.to_period("M") == month+1 and row.exit_date.to_period("M") == month+2
                and row.entry_date < row.exit_date):
            raise ValueError("Benchmark holding dates do not match the configured next-month horizon")
    if not np.isfinite(result[["benchmark_return", "rf_return"]]).all().all():
        raise ValueError("Matched benchmark and RF must be finite decimal returns")
    return result


def performance_metrics(monthly):
    """Relative-field diagnostics always; absolute NAV metrics only when recoverable."""
    monthly = monthly.sort_values("signal_month")
    relative = monthly.net_relative.to_numpy()
    sd = relative.std(ddof=1)
    additive = np.r_[0., relative.cumsum()]
    result = {"months": len(relative), "annual_arithmetic_relative": float(12*relative.mean()),
              "tracking_error": float(np.sqrt(12)*sd),
              "information_ratio": float(np.sqrt(12)*relative.mean()/sd) if sd > 0 else None,
              "additive_relative_drawdown_percentage_points": float(100*np.max(np.maximum.accumulate(additive)-additive)),
              "mean_two_sided_target_turnover": float(monthly.turnover.mean()),
              "mean_monthly_cost_proxy": float(monthly.cost.mean()),
              "positive_relative_month_fraction": float(np.mean(relative > 0))}
    if "net_return" not in monthly:
        return result
    r, rf = monthly.net_return.to_numpy(), monthly.rf_return.to_numpy()
    if np.any(r <= -1):
        raise ValueError("A wealth return at or below-100% is outside this unlevered simulation")
    nav = np.r_[1., np.cumprod(1+r)]
    elapsed = (pd.Timestamp(monthly.exit_date.iloc[-1])-pd.Timestamp(monthly.entry_date.iloc[0])).days
    if elapsed <= 0:
        raise ValueError("Absolute metrics require increasing holding dates")
    excess = r-rf
    excess_sd = excess.std(ddof=1)
    result.update({"final_nav": float(nav[-1]), "cagr_actual_days": float(nav[-1]**(365.25/elapsed)-1),
                   "annual_volatility": float(r.std(ddof=1)*np.sqrt(12)),
                   "sharpe": float(np.sqrt(12)*excess.mean()/excess_sd) if excess_sd > 0 else None,
                   "monthly_boundary_nav_mdd": float(np.max(1-nav/np.maximum.accumulate(nav))),
                   "positive_absolute_month_fraction": float(np.mean(r > 0)),
                   "absolute_metric_scope": "Reconstructed holding-boundary NAV, not observed daily account performance"})
    return result


def backtest(predictions, n=30, weighting="equal", cost_bps=10, risk_feature="std_3m",
             benchmark=None, exposures=None, selected_holdings=None):
    """Use `(model,signal_month,stock)` directly; no stock-only return join.

    Partial stock exposure requires matched benchmark and RF. Costs use target
    weights, omitting holding drift, spread, impact and non-tradable securities.
    """
    if cost_bps < 0:
        raise ValueError("Cost rate cannot be negative")
    if predictions.duplicated(["model", "signal_month", "stock"]).any():
        raise ValueError("Duplicate model + stock + signal_month predictions")
    if selected_holdings is not None and selected_holdings.duplicated(["model", "signal_month", "stock"]).any():
        raise ValueError("Duplicate externally selected holdings")
    benchmark = validate_benchmark(benchmark).set_index("signal_month") if benchmark is not None else None
    monthly, all_holdings = [], []
    for model, model_rows in predictions.groupby("model", sort=True):
        previous = {}
        for month, group in model_rows.groupby("signal_month", sort=True):
            exposure = 1. if exposures is None else float(exposures.loc[month])
            if exposure != 1. and benchmark is None:
                raise ValueError("Cash overlays require absolute benchmark and RF; relative targets alone are insufficient")
            if selected_holdings is None:
                holdings = select_holdings(group, n, weighting, risk_feature, exposure)
            else:
                selected = selected_holdings[selected_holdings.model.eq(model) & selected_holdings.signal_month.eq(month)]
                holdings = selected[["stock", "weight"]].merge(group, on="stock", validate="one_to_one", how="left")
                if (len(holdings) != n or holdings.target.isna().any() or
                        not np.isfinite(holdings.weight).all() or holdings.weight.lt(0).any() or
                        not np.isclose(holdings.weight.sum(), exposure)):
                    raise ValueError("Selected holdings must match current stock-month predictions and exposure")
            current = dict(zip(holdings.stock, holdings.weight))
            turnover = target_turnover(current, previous)
            cost = turnover*cost_bps/10000
            relative = float(holdings.weight @ holdings.target)
            row = {"model": model, "signal_month": month, "stock_exposure": exposure,
                   "turnover": turnover, "cost": cost, "gross_weighted_target": relative,
                   "net_relative": relative-cost}
            if benchmark is not None:
                if month not in benchmark.index:
                    raise ValueError(f"No matched benchmark horizon for {month}")
                b = benchmark.loc[month]
                absolute = relative+exposure*b.benchmark_return+(1-exposure)*b.rf_return-cost
                row.update({"benchmark_return": b.benchmark_return, "rf_return": b.rf_return,
                            "entry_date": b.entry_date, "exit_date": b.exit_date,
                            "net_return": absolute, "net_relative": absolute-b.benchmark_return})
            monthly.append(row)
            all_holdings.append(holdings)
            previous = current
    monthly = pd.DataFrame(monthly)
    metrics = [{"model": model, "n": n, "weighting": weighting, "cost_bps": cost_bps,
                **performance_metrics(g)} for model, g in monthly.groupby("model", sort=True)]
    return monthly, pd.concat(all_holdings, ignore_index=True), pd.DataFrame(metrics)
