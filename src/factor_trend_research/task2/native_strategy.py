"""Causal GMAT-style rules around native_engine; no bundled market-data run.

Call contract
-------------
``risk_returns``: complete, aligned, 16-column CNY exposure-return matrix on
the union valuation calendar, including independently verified holiday carries.
It is used only for portfolio risk, not as a funded-asset accounting shortcut.
``signal_frames``: precomputed/as-of aligned SignalFrames (16-column score,
vol21/22/65/130/252, return/vol correlation, and MA5-confirmed reversal). The
native-observation pipeline computes lookbacks before calendar alignment.
Alternatively ``signal_prices`` builds the old four-trend rules on its supplied
row calendar; that fallback is intended for analytical fixtures, not a claim
that padded holidays constitute native observations.
``market_long``: MultiIndex(date,asset_id), with 20 actual legs and IntervalMark
columns: start_price,end_price,start_fx,end_fx,held_contract,next_contract,
next_price,tradeable. Every executed interval needs all actual marks. Contract
ids refer to the interval's held contract, with same-day next quotes for rolls.
``common_open``: independently declared common-session boolean calendar.
``caps``: 16 logical-asset limits; ``leg_mapping``: 15 singleton mappings and
one five-leg equal-notional black-metals basket. Actual Instrument specs declare
five funded index exposures and fifteen futures legs with native point values.

The caller's native-data gate must explicitly approve real inputs. Fixture runs
are labeled separately. This module returns daily accounting and decisions, not
aggregate portfolio metrics, and never downloads or approves its own data.

Rules are fixed before results: top11; equal risk coefficient .08/11; individual
caps; volatility max22/65/130; >4.5% de-risk toward4%; monthly <3% scale toward4%;
four-trend ranks; MA5 reversal primary; lag>=1. Monthly phase starts only on the
fourth common session when ready and four sessions4--7 fit the declared window.
If day4 is unready, days5--7 do not start a partial new phase.

Synchronous daily valuation/close sizing across CN/US/ICE and same-day FX/VM
conversion are research approximations. They do not assert simultaneous real
close fills. The caller supplies only already-available, lagged signal frames.
Native futures have no imaginary funded principal. No futures margin-call or
collateral-interest model is included. Basket weights are a declared equal-
notional convention because the original report's within-basket mix is unknown.
"""

from dataclasses import dataclass, asdict, replace
from collections.abc import Mapping
import math

import numpy as np
import pandas as pd

from .native_engine import (
    Instrument, IntervalMark, PortfolioState, account_interval, expand_target_weights,
)


from . import rules as _rules
Config = _rules.Config
DEFAULT_CONFIG = replace(Config(), reversal_confirmation="price_above_ma5")
MappingLike = (Mapping, pd.Series)


@dataclass(frozen=True)
class SignalFrames:
    score: pd.DataFrame
    vol: dict[int, pd.DataFrame]
    correlation: pd.DataFrame
    reversal: pd.DataFrame


@dataclass(frozen=True)
class NativeStrategyRun:
    daily: pd.DataFrame
    logical_weights: pd.DataFrame
    actual_positions: pd.DataFrame
    trades: pd.DataFrame
    decisions: list
    final_state: PortfolioState
    protocol: dict


def _aligned(name, panel, index, columns):
    if not isinstance(panel, pd.DataFrame) or not panel.index.equals(index) or set(panel.columns) != set(columns):
        raise ValueError(f"{name} must have the exact valuation index and logical ids")
    panel = panel.loc[:, columns].astype(float)
    if np.isinf(panel.to_numpy()).any():
        raise ValueError(f"{name} contains infinite observations")
    return panel


class PreparedSignals:
    """Adapter for precomputed native-observation/as-of signal frames."""
    target = _rules.Signals.target

    def __init__(self, frames, risk_returns):
        index, columns = risk_returns.index, risk_returns.columns
        self.score = _aligned("score", frames.score, index, columns)
        self.prices = self.score  # Legacy target uses this only for column count.
        self.vol = {n: _aligned(f"vol{n}", frames.vol[n], index, columns)
                    for n in (21, 22, 65, 130, 252)}
        for panel in self.vol.values():
            if (panel < 0).any().any():
                raise ValueError("Volatility frames cannot be negative")
        self.correlation = _aligned("correlation", frames.correlation, index, columns)
        if (self.correlation.abs() > 1 + 1e-10).any().any():
            raise ValueError("Correlation outside [-1,1]")
        reversal = _aligned("reversal", frames.reversal, index, columns)
        values = reversal.to_numpy()
        if not np.isin(values[np.isfinite(values)], [0, 1]).all():
            raise ValueError("Reversal frame must contain boolean/0/1 flags")
        self._reversal_ready = reversal.notna().all(axis=1).to_numpy()
        self.reversal = reversal.fillna(False).astype(bool).to_numpy()
        self.r, self.values = risk_returns, risk_returns.to_numpy()

    def ready(self, i):
        if i < 129 or not self._reversal_ready[i]:
            return False
        if not np.isfinite(self.values[i - 129:i + 1]).all():
            return False
        return (np.isfinite(self.score.iloc[i]).all()
                and all(np.isfinite(panel.iloc[i]).all() for panel in self.vol.values()))


def fixture_signal_frames(prices, risk_returns, cfg=DEFAULT_CONFIG):
    """Old four-trend logic on supplied rows; does not invent missing returns."""
    columns, index = prices.columns, prices.index
    prices = _aligned("signal prices", prices, index, columns)
    if (prices <= 0).any().any():
        raise ValueError("Signal levels must be positive when observed")
    r = _aligned("risk returns", risk_returns, index, columns)
    vol = {n: r.rolling(n, min_periods=n).std() * np.sqrt(cfg.annualization)
           for n in (21, 22, 65, 130, 252)}
    r1, r12 = prices / prices.shift(21) - 1, prices / prices.shift(252) - 1
    low, high = prices.rolling(252, min_periods=252).min(), prices.rolling(252, min_periods=252).max()
    position = ((prices - low) / (high - low).replace(0, np.nan)).mask(high == low, .5)
    components = [r1 / vol[21].clip(lower=cfg.volatility_floor),
                  r12 / vol[252].clip(lower=cfg.volatility_floor), r12, position]
    if cfg.ranking == "percentile_mean":
        components = [x.rank(axis=1, pct=True, method="average") for x in components]
    elif cfg.ranking != "raw_mean":
        raise ValueError("Unknown ranking convention")
    complete_components = np.logical_and.reduce([x.notna().to_numpy() for x in components])
    score = (sum(components) / 4).where(complete_components)
    previous = prices
    alignment = pd.DataFrame(0., index=index, columns=columns)
    for n in range(2, 261):
        current = prices.rolling(n, min_periods=n).mean()
        alignment += (previous >= current).astype(float)
        previous = current
    alignment /= 259
    alignment = alignment.where(prices.rolling(260, min_periods=260).count() == 260)
    valley = np.full(len(columns), np.nan)
    reverse = np.zeros(prices.shape, dtype=bool)
    for i, row in enumerate(alignment.to_numpy()):
        for j, value in enumerate(row):
            if not np.isfinite(value):
                continue
            if value > .35:
                valley[j] = np.nan
                continue
            valley[j] = value if not np.isfinite(valley[j]) else min(valley[j], value)
            reverse[i, j] = valley[j] <= .1 and value >= valley[j] + .005
    if cfg.reversal_confirmation == "price_above_ma5":
        reverse &= (prices > prices.rolling(5, min_periods=5).mean()).to_numpy()
    elif cfg.reversal_confirmation == "five_day_return":
        reverse &= (prices / prices.shift(5) - 1 > 0).to_numpy()
    else:
        raise ValueError("Unknown reversal confirmation")
    reverse_frame = pd.DataFrame(reverse, index=index, columns=columns).where(alignment.notna())
    # A252-observation row lookback is a fixture convention. Production native
    # frames estimate own-session lookbacks before as-of alignment instead.
    correlation = r.rolling(252, min_periods=130).corr(vol[21])
    return SignalFrames(score, vol, correlation, reverse_frame)


def _optional(value):
    return None if pd.isna(value) else value


def _marks_for_date(market_long, d, ids):
    try:
        day = market_long.xs(d, level="date")
    except KeyError as exc:
        raise ValueError(f"Missing actual-leg interval marks on {d.date()}") from exc
    if set(day.index) != set(ids) or day.index.has_duplicates:
        raise ValueError(f"Need each of the20 actual legs exactly once on {d.date()}")
    result = {}
    for a in ids:
        row = day.loc[a]
        mask = row["tradeable"]
        if not isinstance(mask, (bool, np.bool_)):
            raise ValueError("Tradeable must be an explicit boolean; unknown sessions are not holidays")
        result[a] = IntervalMark(
            row["start_price"], row["end_price"], row["start_fx"], row["end_fx"],
            _optional(row["held_contract"]), _optional(row["next_contract"]),
            _optional(row["next_price"]), bool(mask))
    return result


def run_native_strategy(market_long, risk_returns, common_open, caps,
                        instruments, leg_mapping, *, start, end,
                        signal_frames=None, signal_prices=None, cfg=DEFAULT_CONFIG,
                        initial_nav_cny=1_000_000., approved_native_data=False,
                        analytical_fixture=False):
    if not approved_native_data and not analytical_fixture:
        raise ValueError("Real native inputs require the caller's passed data gate")
    if cfg.execution_lag < 1:
        raise ValueError("At least one complete valuation-date signal lag is required")
    if tuple(cfg.rebalance_days) != (4, 5, 6, 7):
        raise ValueError("This wrapper declares the complete four-session4--7 phase")
    if not isinstance(risk_returns.index, pd.DatetimeIndex):
        raise ValueError("Risk returns need a date-only DatetimeIndex")
    dates, logical = risk_returns.index, list(risk_returns.columns)
    if dates.tz is not None or not dates.equals(dates.normalize()) or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Valuation dates must be unique, ascending, date-only and timezone-resolved")
    if len(logical) != 16 or len(set(logical)) != 16:
        raise ValueError("Need exactly16 unique logical asset columns")
    risk_returns = _aligned("risk returns", risk_returns, dates, logical)
    for a in logical:
        first = risk_returns[a].first_valid_index()
        if first is not None and risk_returns.loc[first:, a].isna().any():
            raise ValueError("Unknown internal risk-return gaps cannot be filled or skipped")
    if not isinstance(common_open, pd.Series) or not common_open.index.equals(dates) or common_open.isna().any():
        raise ValueError("Common-open calendar must be complete and aligned")
    if not all(isinstance(v, (bool, np.bool_)) for v in common_open):
        raise ValueError("Common-open values must be explicit booleans")
    if set(leg_mapping) != set(logical):
        raise ValueError("Leg mapping must cover the16 logical assets")
    specs = {x.asset_id: x for x in instruments}
    ids = list(specs)
    if len(ids) != 20 or len(instruments) != 20 or sum(x.kind == "funded" for x in instruments) != 5:
        raise ValueError("Need five funded indices and fifteen actual futures legs")
    if any(abs(x.one_way_cost_rate - cfg.transaction_cost) > 1e-12 for x in instruments):
        raise ValueError("Instrument cost rates must match the predeclared strategy cost rate")
    flattened = [a for legs in leg_mapping.values() for a in legs]
    if len(flattened) != 20 or set(flattened) != set(ids) or len(set(flattened)) != 20:
        raise ValueError("Actual legs must be mapped once each, with no overlap")
    baskets = [g for g, legs in leg_mapping.items() if len(legs) > 1]
    if len(baskets) != 1 or len(leg_mapping[baskets[0]]) != 5:
        raise ValueError("Declare exactly one five-leg black-metals basket")
    basket = baskets[0]
    basket_legs = list(leg_mapping[basket])
    if any(specs[a].kind != "future" for a in basket_legs) or any(
            abs(v - .2) > 1e-12 for v in leg_mapping[basket].values()):
        raise ValueError("The five-leg basket uses predeclared equal-notional future legs")
    expand_target_weights(dict.fromkeys(logical, 0), leg_mapping, instruments)
    if isinstance(caps, MappingLike):
        if set(caps) != set(logical):
            raise ValueError("Caps must cover the16 logical assets")
        caps = np.asarray([caps[a] for a in logical], dtype=float)
    else:
        caps = np.asarray(caps, dtype=float)
    if caps.shape != (16,) or not np.isfinite(caps).all() or (caps < 0).any():
        raise ValueError("Need16 finite nonnegative logical caps")
    if not isinstance(market_long.index, pd.MultiIndex) or list(market_long.index.names) != ["date", "asset_id"]:
        raise ValueError("market_long must use a(date,asset_id) MultiIndex")
    required = {"start_price", "end_price", "start_fx", "end_fx", "held_contract", "next_contract", "next_price", "tradeable"}
    if not required.issubset(market_long.columns):
        raise ValueError("market_long lacks IntervalMark fields")
    if (signal_frames is None) == (signal_prices is None):
        raise ValueError("Supply exactly one of precomputed signal_frames or fixture signal_prices")
    if signal_frames is None and not analytical_fixture:
        raise ValueError("Real runs require native-observation signal_frames; row-calendar fallback is fixture-only")
    source = "precomputed_native_observations_asof" if signal_frames is not None else "supplied_row_calendar_fallback"
    if signal_frames is None:
        signal_prices = _aligned("signal prices", signal_prices, dates, logical)
        signal_frames = fixture_signal_frames(signal_prices, risk_returns, cfg)
    signals = PreparedSignals(signal_frames, risk_returns)
    run_ids = np.flatnonzero((dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end)))
    if len(run_ids) == 0 or run_ids[0] == 0:
        raise ValueError("Need a nonempty run window plus a preceding valuation date")
    if not np.isfinite(initial_nav_cny) or initial_nav_cny <= 0:
        raise ValueError("Initial NAV must be positive")
    # Calendar knowledge is declared independently of price outcomes. No future
    # signal values or next-month returns are consulted when scheduling phases.
    ordinals, phases = {}, {}
    common_mask = common_open.to_numpy(dtype=bool)
    for month, group in pd.Series(dates[common_mask], index=dates[common_mask]).groupby(
            lambda d: (d.year, d.month)):
        phase = []
        for k, d in enumerate(group, 1):
            ordinals[d] = k
            if k in cfg.rebalance_days:
                phase.append(d)
        phases[month] = phase
    state = PortfolioState(float(initial_nav_cny), as_of=dates[run_ids[0] - 1].date())
    daily_rows, weight_rows, position_rows, trade_rows, decisions = [], [], [], [], []
    phase_target, phase_remaining, phase_month = None, 0, None
    for i in run_ids:
        d, previous_date = dates[i], dates[i - 1]
        decision_i = i - cfg.execution_lag
        marks = _marks_for_date(market_long, d, ids)
        common = bool(common_open.iloc[i])
        if common and not all(m.tradeable for m in marks.values()):
            raise ValueError("Common-open calendar contradicts actual-leg tradeable masks")
        # Mark-only preview has no trades or contract changes. The original
        # state is used for the one actual accounting execution below.
        frozen_marks = {a: replace(m, next_contract=None, next_price=None, tradeable=False)
                        for a, m in marks.items()}
        preview = account_interval(state, instruments, frozen_marks,
                                   start_date=previous_date.date(), end_date=d.date())
        management_fee = preview.end_nav_cny * (1 - math.exp(
            -cfg.management_fee * (d - previous_date).days / 365.25))
        pretrade_nav = preview.end_nav_cny - management_fee
        if pretrade_nav <= 0:
            raise ValueError("Management fee exhausts NAV")
        exposures = {**preview.funded_market_value_cny, **preview.futures_notional_cny}
        before = np.array([sum(exposures[a] for a in leg_mapping[g]) / pretrade_nav for g in logical])
        candidate = before.copy()
        k = ordinals.get(d)
        month = (d.year, d.month)
        ready = decision_i >= 0 and signals.ready(decision_i)
        scheduled, reversal_count = False, 0
        if k == 4:
            phase = phases.get(month, [])
            complete = len(phase) == 4 and phase[0] >= dates[run_ids[0]] and phase[-1] <= dates[run_ids[-1]]
            phase_target, phase_remaining, phase_month = None, 0, None
            if ready and complete:
                phase_target, chosen, reversal_count = signals.target(decision_i, caps, cfg)
                phase_remaining, phase_month = 4, month
                decisions.append({"event": "monthly_phase_start", "execution_date": str(d.date()),
                                  "signal_date": str(dates[decision_i].date()),
                                  "selected": [logical[j] for j in chosen],
                                  "target": phase_target.tolist(), "reversal_boosted_assets": reversal_count})
            else:
                decisions.append({"event": "monthly_phase_skipped", "execution_date": str(d.date()),
                                  "reason": "unready_on_day4" if not ready else "incomplete_four_day_window"})
        if k in cfg.rebalance_days and phase_month == month and phase_remaining > 0:
            candidate = before + (phase_target - before) / phase_remaining
            phase_remaining -= 1
            scheduled = True
        candidate = _rules.project_weights(candidate, caps, cfg.max_risky_exposure)
        risk_estimate, risk_trigger = np.nan, False
        if cfg.daily_vol_control and ready:
            candidate, risk_estimate, risk_trigger = _rules.risk_scale(
                signals.values, decision_i, candidate, caps, cfg)
        changed = np.max(np.abs(candidate - before)) > 1e-12
        target_actual, basket_fixed = None, None
        allocation_trade = common and changed
        if allocation_trade:
            target_actual = expand_target_weights(dict(zip(logical, candidate)), leg_mapping, instruments)
        elif all(marks[a].tradeable for a in basket_legs):
            basket_total = sum(exposures[a] for a in basket_legs)
            basket_fixed = {a: basket_total / 5 for a in basket_legs}
        result = account_interval(
            state, instruments, marks, start_date=previous_date.date(), end_date=d.date(),
            target_weights=target_actual, fixed_notional_targets_cny=basket_fixed,
            management_fee_cny=management_fee, roll_policy="notional",
            target_gross_limit=cfg.max_risky_exposure)
        after_exposures = {**result.funded_market_value_cny, **result.futures_notional_cny}
        logical_after = {g: sum(after_exposures[a] for a in leg_mapping[g]) / result.end_nav_cny for g in logical}
        daily_rows.append({"date": d, "nav": result.end_nav_cny,
                           "return": result.end_nav_cny / result.opening_nav_cny - 1,
                           "opening_nav_cny": result.opening_nav_cny,
                           "funded_pnl_cny": sum(result.funded_pnl_cny.values()),
                           "futures_vm_cny": sum(result.futures_vm_cny.values()),
                           "management_fee_cny": management_fee,
                           "transaction_cost_cny": result.transaction_cost_cny,
                           "traded_notional_cny": result.gross_traded_notional_cny,
                           "cash_cny": result.state.cash_cny, "cash_weight": result.cash_weight,
                           "funded_value_cny": sum(result.funded_market_value_cny.values()),
                           "futures_gross_notional_cny": sum(result.futures_notional_cny.values()),
                           "gross_risky_exposure": sum(logical_after.values()),
                           "common_open": common, "signals_ready": ready,
                           "scheduled_rebalance": scheduled, "allocation_trade": allocation_trade,
                           "allocation_signal_deferred": bool(changed and not common),
                           "phase_remaining": phase_remaining if phase_month == month else 0,
                           "risk_reset": bool(risk_trigger and allocation_trade),
                           "risk_signal_deferred": bool(risk_trigger and not common),
                           "estimated_volatility": risk_estimate,
                           "deferred_actual_legs": ",".join(result.deferred)})
        weight_rows.append({"date": d, **logical_after})
        for a in ids:
            is_future = specs[a].kind == "future"
            position_rows.append({"date": d, "asset_id": a, "kind": specs[a].kind,
                                  "contract": result.state.future_contracts.get(a),
                                  "quantity": (result.state.future_quantities if is_future else result.state.funded_units)[a],
                                  "mark_price": result.state.last_marks[a].price,
                                  "fx": result.state.last_marks[a].fx,
                                  "funded_value_cny": result.funded_market_value_cny.get(a, 0),
                                  "futures_notional_cny": result.futures_notional_cny.get(a, 0)})
        for trade in result.trades:
            trade_rows.append({"date": d, **asdict(trade)})
        state = result.state
    protocol = {"config": asdict(cfg), "signal_source": source,
                "input_kind": "analytical_fixture" if analytical_fixture else "caller_gated_native_inputs",
                "caller_native_data_gate_passed": bool(approved_native_data),
                "logical_assets": logical, "actual_legs": ids, "leg_mapping": leg_mapping,
                "instrument_specs": [asdict(s) for s in instruments],
                "basket_internal_rebalance": "daily equal notional; preserve pre-fee group notional unless overall allocation changes",
                "futures_accounting": "held-contract point P&L; immediate end-FX conversion to CNY VM; no principal FX return",
                "execution": "lagged decisions; synchronous research EOD sizing; new positions earn from next interval",
                "not_modeled": ["integer lots", "margin calls", "collateral interest", "spread/market impact", "real asynchronous fills"],
                "aggregate_metrics_computed": False}
    return NativeStrategyRun(pd.DataFrame(daily_rows).set_index("date"),
                             pd.DataFrame(weight_rows).set_index("date"),
                             pd.DataFrame(position_rows), pd.DataFrame(trade_rows, columns=[
                                 "date", "asset_id", "kind", "action", "contract", "quantity_delta",
                                 "price", "point_value", "fx", "gross_notional_cny", "cost_cny"]),
                             decisions, state, protocol)
