"""Causal, separately labeled portfolio extensions; no test-based rule selection."""
import numpy as np
import pandas as pd
from .portfolio import select_holdings


def market_exposure(prices, signal_months, rule="trend_sma10"):
    """Completed history through signal-month-end only, before next-month entry.

    prices is a daily `date,close` frame for one benchmark. A 15% reference
    scales benchmark realized volatility, not a guarantee of portfolio volatility.
    """
    data = prices.copy()
    data["date"] = pd.to_datetime(data.date)
    data = data.sort_values("date")
    if data.date.duplicated().any() or not np.isfinite(data.close).all() or data.close.le(0).any():
        raise ValueError("Market prices require unique dates and finite positive closes")
    result = []
    for month in sorted(signal_months):
        history = data[data.date.le(pd.Period(month, "M").end_time)]
        monthly = history.groupby(history.date.dt.to_period("M")).close.last()
        if monthly.empty or str(monthly.index[-1]) != month:
            raise ValueError(f"Missing completed signal-month market history for {month}")
        if rule in ["trend_sma10", "trend_sma10_vol63_target15"] and len(monthly) < 10:
            raise ValueError("SMA10 needs ten observed monthly closes")
        if rule == "momentum12" and len(monthly) < 13:
            raise ValueError("Twelve-month momentum needs thirteen monthly closes")
        if rule in ["vol63_target15", "trend_sma10_vol63_target15"] and len(history) < 64:
            raise ValueError("63daily returns require at least64daily closes")
        if rule == "trend_sma10":
            g = float(monthly.iloc[-1] > monthly.iloc[-10:].mean())
        elif rule == "momentum12":
            g = float(monthly.iloc[-1]/monthly.iloc[-13]-1 > 0)
        elif rule in ["vol63_target15", "trend_sma10_vol63_target15"]:
            vol = history.close.pct_change().dropna().iloc[-63:].std(ddof=1)*np.sqrt(252)
            g = min(1., .15/vol) if vol > 0 else 1.
            if rule == "trend_sma10_vol63_target15":
                g *= float(monthly.iloc[-1] > monthly.iloc[-10:].mean())
        else:
            raise ValueError("Unknown market regime rule")
        result.append({"signal_month": month, "stock_exposure": g,
                       "max_market_data_date": history.date.max()})
    return pd.DataFrame(result)


def relative_defensive_holdings(predictions, universe_panel, lookback=12, candidate_pool=100,
                               n=30, risk_feature="std_3m", training_gap=2, always_defensive=False):
    """Old relative-state extension: change the stock basket, remain fully invested.

    Weakness=sum(last known6/12universe equal-weight relative returns)<0.
    Only target signal months <= current signal month minus2 are observable.
    In weakness choose the lowest-risk-feature30 names among score Top100/200.
    """
    if lookback not in [6, 12] or candidate_pool not in [100, 200] or training_gap < 2:
        raise ValueError("Use the prespecified6/12month and100/200candidate rules with gap>=2")
    pool = universe_panel.groupby("signal_month").target.mean().sort_index()
    holdings, states = [], []
    for (model, month), group in predictions.groupby(["model", "signal_month"], sort=True):
        cutoff = str(pd.Period(month, "M")-training_gap)
        required = [str(pd.Period(cutoff, "M")-i) for i in range(lookback-1, -1, -1)]
        known = pool.reindex(required)
        if known.isna().any():
            raise ValueError("Insufficient realized history for the relative-state window")
        weak = bool(known.sum() < 0)
        if weak or always_defensive:
            if candidate_pool > len(group) or risk_feature not in group or not np.isfinite(group[risk_feature]).all():
                raise ValueError("Defensive selection needs a complete risk feature and enough candidate names")
            candidates = group.sort_values(["score", "stock"], ascending=[False, True], kind="stable").head(candidate_pool)
            selected = candidates.sort_values([risk_feature, "score", "stock"],
                ascending=[True, False, True], kind="stable").head(n).copy()
            selected["weight"] = 1/n
        else:
            selected = select_holdings(group, n=n)
        holdings.append(selected)
        states.append({"model": model, "signal_month": month, "weak_state": weak,
                       "defensive_active": bool(weak or always_defensive),
                       "last_available_target_signal": known.index[-1], "historical_pool_relative_sum": known.sum()})
    return pd.concat(holdings, ignore_index=True), pd.DataFrame(states)
