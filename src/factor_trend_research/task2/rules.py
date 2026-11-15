"""Causal, cash-aware GMAT-style research engine; not an index replication.

Signals observed on t-1 are executed at the modelled close on t. New holdings
earn returns only from t to t+1. All prices/returns must be independently
validated by the acquisition pipeline. No optimiser fits these fixed rules.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    top_n: int = 11
    annualization: int = 252
    risk_budget: float = 0.08
    vol_target: float = 0.04
    vol_upper: float = 0.045
    vol_lower: float = 0.03
    gross_limit: float = 1.0
    cash_buffer: float = 0.005
    volatility_floor: float = 0.005
    transaction_cost: float = 0.0005
    management_fee: float = 0.005
    ranking: str = "percentile_mean"
    mode: str = "momentum"
    reversal: bool = True
    reversal_confirmation: str = "five_day_return"
    reversal_increment: float = 0.05
    adaptive_volatility: bool = True
    daily_vol_control: bool = True
    monthly_vol_target: bool = True
    execution_lag: int = 1
    rebalance_days: tuple = (4, 5, 6, 7)

    @property
    def max_risky_exposure(self):
        return self.gross_limit - self.cash_buffer


def project_weights(weights, caps, gross_limit=1.0):
    """Preserve residual cash; never re-normalise capped risk holdings to one."""
    w = np.minimum(np.maximum(np.asarray(weights, float), 0), caps)
    if w.sum() > gross_limit:
        w *= gross_limit / w.sum()
    return w


def portfolio_volatility(returns, i, weights, annualization=252):
    """Risk under current hypothetical holdings, using past observations only."""
    values = []
    for window in (22, 65, 130):
        if i + 1 >= window:
            path = returns[i + 1 - window:i + 1] @ weights
            values.append(float(np.std(path, ddof=1) * np.sqrt(annualization)))
    return max(values) if values else np.nan


def risk_scale(returns, i, w, caps, cfg, allow_increase=False):
    vol = portfolio_volatility(returns, i, w, cfg.annualization)
    triggered = False
    if np.isfinite(vol) and vol > cfg.vol_upper:
        w = w * cfg.vol_target / vol
        triggered = True
    elif allow_increase and np.isfinite(vol) and 0 < vol < cfg.vol_lower:
        w = project_weights(w * cfg.vol_target / vol, caps, cfg.max_risky_exposure)
    return project_weights(w, caps, cfg.max_risky_exposure), vol, triggered


class Signals:
    """Target allocator mixed into prepared native signal frames."""
    def target(self, i, caps, cfg):
        n = self.prices.shape[1]
        score = self.score.iloc[i].to_numpy()
        if not np.isfinite(score).all():
            raise ValueError("Insufficient pre-decision signal history")
        if cfg.mode in ("equal_all", "inverse_vol_all"):
            chosen = np.arange(n)
        else:
            # Stable ordering makes score ties deterministic.
            chosen = np.argsort(-score, kind="stable")[:cfg.top_n]
        if cfg.mode == "equal_all":
            w = np.zeros(n)
            w[chosen] = 1 / len(chosen)
        else:
            conservative = np.maximum.reduce([self.vol[k].iloc[i].to_numpy()
                                               for k in (22, 65, 130)])
            sigma = conservative.copy()
            if cfg.adaptive_volatility and cfg.mode == "momentum":
                corr = self.correlation.iloc[i].to_numpy()
                sigma = np.where(corr < 0, self.vol[21].iloc[i].to_numpy(), sigma)
            sigma = np.maximum(sigma, cfg.volatility_floor)
            w = np.zeros(n)
            w[chosen] = (cfg.risk_budget / len(chosen)) / sigma[chosen]
        w = project_weights(w, caps, cfg.max_risky_exposure)
        reversal_count = 0
        if cfg.reversal and cfg.mode == "momentum":
            active = np.intersect1d(chosen, np.flatnonzero(self.reversal[i]))
            # Allocate a predeclared five percentage points at most to each
            # triggered asset, funded proportionally from other risk holdings.
            # Preserve gross exposure; no order-dependent sequential boosting.
            additions = np.zeros(n)
            additions[active] = np.minimum(cfg.reversal_increment,
                                           np.maximum(caps[active] - w[active], 0))
            donors = np.ones(n, bool)
            donors[active] = False
            available = w[donors].sum()
            total = additions.sum()
            if total > 0 and available > 0:
                additions *= min(1., available / total)
                w[donors] *= 1 - additions.sum() / available
                w += additions
                reversal_count = int((additions > 0).sum())
        w = project_weights(w, caps, cfg.max_risky_exposure)
        if cfg.monthly_vol_target:
            w, _, _ = risk_scale(self.values, i, w, caps, cfg, allow_increase=True)
        return w, chosen, reversal_count
