"""Portable orchestration and reconciled output for Task2."""
from dataclasses import fields, replace
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .native_signals import build_signals
from .native_strategy import DEFAULT_CONFIG, run_native_strategy
from .universe import LOGICAL, CAPS, instruments


def configuration(path=None):
    cfg = DEFAULT_CONFIG
    caps = dict(CAPS)
    if path is not None:
        obj = json.loads(Path(path).read_text())
        rules = obj.get("rules", {})
        if set(rules) - {f.name for f in fields(cfg)}:
            raise ValueError("Unknown strategy configuration key")
        if "rebalance_days" in rules:
            rules["rebalance_days"] = tuple(rules["rebalance_days"])
        cfg = replace(cfg, **rules)
        caps = obj.get("caps", caps)
    if cfg.annualization != 252 or cfg.reversal_confirmation != "price_above_ma5":
        raise ValueError("Native signal implementation declares252annualization andMA5 confirmation")
    if cfg.ranking != "percentile_mean":
        raise ValueError("Native signals declare four-component mean-percentile ranking")
    if cfg.volatility_floor != 0.005:
        raise ValueError("Native signal ranking declares the fixed0.005volatility floor")
    if cfg.mode not in ("momentum", "equal_all", "inverse_vol_all"):
        raise ValueError("Unknown allocation mode")
    for name in ("reversal", "adaptive_volatility", "daily_vol_control", "monthly_vol_target"):
        if not isinstance(getattr(cfg, name), bool):
            raise ValueError(f"{name} must be a JSON boolean")
    for name in ("risk_budget", "vol_target", "vol_upper", "vol_lower", "gross_limit",
                 "cash_buffer", "volatility_floor", "transaction_cost", "management_fee", "reversal_increment"):
        if not np.isfinite(getattr(cfg, name)):
            raise ValueError(f"{name} must be finite")
    if not 1 <= cfg.top_n <= 16 or cfg.execution_lag < 1:
        raise ValueError("Require1..16selected assets and positive execution lag")
    if not (0 < cfg.vol_lower < cfg.vol_target < cfg.vol_upper):
        raise ValueError("Require0<vol_lower<vol_target<vol_upper")
    if not 0 <= cfg.transaction_cost < 1 or cfg.management_fee < 0:
        raise ValueError("Invalid transaction or management fee")
    if not 0 < cfg.max_risky_exposure <= 1:
        raise ValueError("This version is unlevered and keeps a cash buffer")
    if cfg.risk_budget <= 0 or cfg.volatility_floor <= 0 or cfg.reversal_increment < 0:
        raise ValueError("Risk budget/floor must be positive and reversal increment nonnegative")
    if set(caps) != set(LOGICAL) or not all(np.isfinite(x) and 0 <= x <= 1 for x in caps.values()):
        raise ValueError("Need one valid cap for each16logical exposure")
    return cfg, caps


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(x) for x in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def metrics(daily, initial_date, initial_nav=1_000_000):
    r = daily["return"].to_numpy(dtype=float)
    nav = np.r_[1.0, daily.nav.to_numpy() / initial_nav]
    years = (daily.index[-1] - pd.Timestamp(initial_date)).days / 365.25
    if years <= 0 or np.any(nav <= 0):
        raise ValueError("Need positive NAV and positive elapsed time")
    sd = np.std(r, ddof=1)
    return dict(observations=len(r), initial_date=str(pd.Timestamp(initial_date).date()),
        first_return_date=str(daily.index[0].date()), end_date=str(daily.index[-1].date()),
        elapsed_years=years, total_return=float(nav[-1] - 1),
        cagr=float(nav[-1] ** (1 / years) - 1), annualized_volatility=float(sd * np.sqrt(252)),
        sharpe_rf_zero=float(np.mean(r) / sd * np.sqrt(252)) if sd > 0 else None,
        maximum_drawdown=float(np.max(1 - nav / np.maximum.accumulate(nav))),
        transaction_cost_cny=float(daily.transaction_cost_cny.sum()),
        management_fee_cny=float(daily.management_fee_cny.sum()))


def run(data, output, *, config_path=None, initial_nav=1_000_000):
    cfg, caps = configuration(config_path)
    frames, logical_risk, counts, _, signal_status = build_signals(
        data.local_returns, data.risk_returns, data.observed)
    signal_status.pop("full16_run_performed", None)
    signal_status["signal_preparation_only"] = True
    start = pd.Timestamp(data.metadata["evaluation_start"])
    end = pd.Timestamp(data.metadata["evaluation_end"])
    if start not in logical_risk.index or end not in logical_risk.index or end < start:
        raise ValueError("Declared evaluation start/end must be actual valuation dates")
    # A real-input provider supplies a complete, audited warm-up window. Unknown
    # risk intervals are not zero-filled or skipped by automatically picking a
    # favorable suffix. The native strategy rejects internal missing risk data.
    synthetic = data.metadata.get("input_kind") == "synthetic_demo"
    result = run_native_strategy(data.market, logical_risk, data.common_open, caps,
        instruments(cfg.transaction_cost), LOGICAL, start=start, end=end,
        signal_frames=frames, cfg=cfg, initial_nav_cny=initial_nav,
        approved_native_data=not synthetic and data.metadata.get("data_gate_passed") is True,
        analytical_fixture=synthetic)
    first_i = logical_risk.index.get_loc(result.daily.index[0])
    initial_date = logical_risk.index[first_i - 1]
    daily = result.daily
    before = daily.nav.shift(1)
    before.iloc[0] = initial_nav
    delta = daily.funded_pnl_cny + daily.futures_vm_cny - daily.management_fee_cny - daily.transaction_cost_cny
    if not np.allclose(daily.nav - before, delta, rtol=1e-10, atol=1e-5):
        raise AssertionError("Cash/VM/fee ledger does not reconcile to NAV")
    if not np.allclose(daily.nav, daily.cash_cny + daily.funded_value_cny, rtol=1e-10, atol=1e-5):
        raise AssertionError("NAV is cash plus funded assets, not futures notional")
    summary = dict(mode="synthetic_demo" if synthetic else "user_supplied_native_inputs",
        synthetic=synthetic, historical_performance_claim=False if synthetic else "conditional_on_user_supplied_inputs",
        official_reference_replication_claim=False,
        full16_strategy_run=True, full16_run_is_synthetic=synthetic,
        description=("SYNTHETIC SOFTWARE FIXTURE ONLY: generated prices/FX/contracts; not historical investment results."
                     if synthetic else "Local original-underlying research inputs; caller gate is not independent vendor/PIT/execution certification."),
        metrics=metrics(daily, initial_date, initial_nav),
        logical_asset_count=16, actual_leg_count=20, input_metadata=data.metadata,
        signal_preparation=signal_status, strategy_protocol=result.protocol,
        checks=dict(nav_equals_cash_plus_funded=True, nav_flow_reconciliation=True),
        not_modeled=["integer contract lots", "margin calls and collateral interest", "bid/ask spread and market impact",
                     "real asynchronous close execution", "certified point-in-time input releases"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output / "daily_nav_and_cash.csv", index_label="date")
    result.logical_weights.to_csv(output / "logical_exposures.csv", index_label="date")
    result.actual_positions.to_csv(output / "actual_positions.csv", index=False)
    result.trades.to_csv(output / "trades_and_costs.csv", index=False)
    counts.to_csv(output / "native_observation_counts.csv", index_label="date")
    (output / "decisions.json").write_text(json.dumps(_clean(result.decisions), indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(_clean(summary), indent=2, allow_nan=False) + "\n")
    return result, summary
