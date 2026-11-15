"""Deterministic SYNTHETIC16-exposure/20-leg fixtures, not market history.

Artificial weekday sessions, prices, FX and contract rolls exercise software.
No output from this module is evidence of investment performance.
"""
import numpy as np
import pandas as pd

from .inputs import ResearchInputs, validate_inputs
from .universe import FUNDED, FUTURES, USD, LOGICAL


def synthetic_inputs(seed=20261114, periods=520):
    if periods < 360:
        raise ValueError("Synthetic fixture needs at least360 dates for warm-up and a phase")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=periods)
    ids = list(FUNDED + FUTURES)
    common = rng.normal(0, 0.004, periods)
    # Distinct and changing trends; these are generated data, not fitted history.
    r = np.empty((periods, len(ids)))
    for j in range(len(ids)):
        volatility = 0.002 if ids[j] in ("TS", "TF", "T", "ZT", "ZF", "ZN") else 0.008
        drift = 0.0001 + 0.00025 * np.sin(np.arange(periods) / 90 + j)
        r[:, j] = drift + 0.3 * common + rng.normal(0, volatility, periods)
    r = np.clip(r, -0.08, 0.08)
    fx = 7 * np.exp(np.cumsum(rng.normal(0, 0.0008, periods)))
    local = pd.DataFrame(np.nan, index=dates, columns=ids)
    risk = local.copy()
    observed = pd.DataFrame(True, index=dates, columns=ids)
    marks = []
    for j, a in enumerate(ids):
        previous = 100 + 5 * j
        held = f"{a}_SYNTH_0" if a in FUTURES else None
        number = 0
        for i in range(1, periods):
            before_fx = fx[i - 1] if a in USD else 1.0
            after_fx = fx[i] if a in USD else 1.0
            end = previous * (1 + r[i, j])
            roll = a in FUTURES and i % 42 == 0
            next_id, next_price = None, np.nan
            if roll:
                number += 1
                next_id = f"{a}_SYNTH_{number}"
                # A deliberate delivery-price gap; it must NOT become roll P&L.
                next_price = end * (1.02 if j % 2 == 0 else 0.98)
            marks.append(dict(date=dates[i], asset_id=a, start_price=previous, end_price=end,
                start_fx=before_fx, end_fx=after_fx, held_contract=held,
                next_contract=next_id, next_price=next_price, tradeable=True))
            local.loc[dates[i], a] = end / previous - 1
            risk.loc[dates[i], a] = ((end / previous) * (after_fx / before_fx) - 1
                                    if a in FUNDED else local.loc[dates[i], a] * after_fx / before_fx)
            previous = float(next_price) if roll else end
            if roll:
                held = next_id
    market = pd.DataFrame(marks).set_index(["date", "asset_id"]).sort_index()
    metadata = dict(schema_version=1, input_kind="synthetic_demo", seed=int(seed),
        evaluation_start=str(dates[300].date()), evaluation_end=str(dates[-1].date()),
        logical_asset_count=len(LOGICAL), actual_leg_count=len(ids),
        data_gate_passed=False, historical_performance_claim=False,
        calendar_policy="Artificial weekdays: all20 synthetic legs observed/tradeable; not exchange calendars.",
        fx_policy="Generated CNY perUSD path; no real FX source.",
        contract_roll_policy="Synthetic42-session rolls with simultaneous old/new quotes.")
    return validate_inputs(ResearchInputs(market, local, risk, observed,
                                        pd.Series(True, index=dates, name="common_open"), metadata))
