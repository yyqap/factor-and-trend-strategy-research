"""Strict local input contract. No downloads, ETF substitutions or zero filling."""
from dataclasses import dataclass
from pathlib import Path
from io import BytesIO
import hashlib
import json

import numpy as np
import pandas as pd

from .universe import FUNDED, FUTURES, USD, POINTS

FILES = ("accounting_intervals.csv", "local_actual_leg_returns.csv",
         "cny_actual_leg_risk_returns.csv", "native_observed.csv", "common_execution_dates.csv")


@dataclass(frozen=True)
class ResearchInputs:
    market: pd.DataFrame
    local_returns: pd.DataFrame
    risk_returns: pd.DataFrame
    observed: pd.DataFrame
    common_open: pd.Series
    metadata: dict


def _dates(frame, name):
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{name}: expected date index")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name}: dates must be unique and increasing")
    if frame.index.tz is not None or not frame.index.equals(frame.index.normalize()):
        raise ValueError(f"{name}: supply already resolved date-only sessions")


def _booleans(frame, name):
    out = frame.copy()
    for c in frame:
        values = frame[c].astype(str).str.lower().map({"true": True, "false": False})
        if values.isna().any():
            raise ValueError(f"{name}: explicit true/false values required, no unknown sessions")
        out[c] = values.astype(bool)
    return out


def validate_inputs(data):
    """Validate shape, observed masks and quote fields; no vendor certification."""
    ids = list(FUNDED + FUTURES)
    index = data.local_returns.index
    for name, frame in (("local returns", data.local_returns), ("risk returns", data.risk_returns),
                        ("native observed", data.observed)):
        _dates(frame, name)
        if not frame.index.equals(index) or set(frame.columns) != set(ids):
            raise ValueError(f"{name}: require identical dates and all20 original legs")
    if not data.common_open.index.equals(index) or data.common_open.isna().any():
        raise ValueError("Common calendar must cover the full valuation index")
    for name, frame in (("local returns", data.local_returns), ("risk returns", data.risk_returns)):
        values = frame.to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError(f"{name}: infinite returns are invalid")
    local_values = data.local_returns.to_numpy(dtype=float)
    if (local_values[np.isfinite(local_values)] <= -1).any():
        raise ValueError("Local signal returns must be above-100%; no ambiguous nonpositive-price sizing")
    if not all(isinstance(x, (bool, np.bool_)) for x in data.observed.to_numpy().ravel()):
        raise ValueError("Native observations require explicit booleans")
    if not all(isinstance(x, (bool, np.bool_)) for x in data.common_open):
        raise ValueError("Common calendar requires explicit booleans")
    market = data.market
    if list(market.index.names) != ["date", "asset_id"] or market.index.has_duplicates:
        raise ValueError("Market rows need unique(date,asset_id) keys")
    required = {"start_price", "end_price", "start_fx", "end_fx", "held_contract", "next_contract", "next_price", "tradeable"}
    if not required.issubset(market):
        raise ValueError("Missing accounting interval fields")
    # Every interval after the first normalization date must contain20 actual legs.
    mask = market.tradeable.unstack("asset_id").reindex(index=index[1:], columns=ids)
    if mask.isna().any().any() or set(market.index.get_level_values("asset_id")) != set(ids):
        raise ValueError("Incomplete actual-leg intervals; do not invent holiday quotes")
    if not all(isinstance(x, (bool, np.bool_)) for x in mask.to_numpy().ravel()):
        raise ValueError("Interval tradeable masks must be booleans")
    if not data.common_open.loc[index[1:]].eq(mask.all(axis=1)).all():
        raise ValueError("Common calendar disagrees with actual-leg trading masks")
    for name in ("start_price", "end_price", "start_fx", "end_fx"):
        values = market[name].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"{name}: need actual finite positive marks, not imputed gaps")
    for a in FUTURES:
        rows = market.xs(a, level="asset_id")
        if rows.held_contract.isna().any() or rows.held_contract.astype(str).str.len().eq(0).any():
            raise ValueError(f"{a}: missing held native contract IDs")
    for a in FUNDED:
        rows = market.xs(a, level="asset_id")
        if rows[["held_contract", "next_contract", "next_price"]].notna().any().any():
            raise ValueError(f"{a}: modeled funded index cannot contain futures fields")
    for a in ids:
        rows = market.xs(a, level="asset_id").reindex(index[1:])
        expected_local = rows.end_price / rows.start_price - 1
        ratio_fx = rows.end_fx / rows.start_fx
        expected_risk = ((1 + expected_local) * ratio_fx - 1 if a in FUNDED
                         else expected_local * ratio_fx)
        if not np.allclose(data.local_returns.loc[index[1:], a], expected_local, atol=1e-10, rtol=1e-9):
            raise ValueError(f"{a}: local returns disagree with same-held-contract marks")
        if not np.allclose(data.risk_returns.loc[index[1:], a], expected_risk, atol=1e-10, rtol=1e-9):
            raise ValueError(f"{a}: risk returns disagree with funded/VM currency accounting")
        if a not in USD and not (rows.start_fx.eq(1) & rows.end_fx.eq(1)).all():
            raise ValueError(f"{a}: native CNY marks require FX=1")
    return data


def load_native_inputs(directory):
    """Load caller-owned prepared native data, with a declared/hashes-locked gate.

    A user-supplied gate is not independent source/PIT validation. Provenance,
    exchange sessions and contract quote units must be reviewed separately.
    """
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("schema_version") != 1 or metadata.get("input_kind") != "native_original_underlyings":
        raise ValueError("Native metadata must declare schema1/native_original_underlyings")
    if metadata.get("data_gate_passed") is not True:
        raise ValueError("Local native inputs require an explicit passed data gate")
    for key in ("calendar_policy", "fx_policy", "contract_roll_policy", "evaluation_start", "evaluation_end"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"Native metadata needs {key}")
    specs = metadata.get("instruments", [])
    declared = {x.get("asset_id"): x for x in specs}
    if len(specs) != 20 or set(declared) != set(FUNDED + FUTURES):
        raise ValueError("Declare all20 original legs; proxy ETFs cannot replace futures silently")
    for a, spec in declared.items():
        expected_kind = "funded" if a in FUNDED else "future"
        representation = "price_index" if a in FUNDED else "native_futures_contract"
        if spec.get("kind") != expected_kind or spec.get("representation") != representation:
            raise ValueError(f"{a}: wrong representation; no silent ETF/futures substitution")
        if spec.get("currency") != ("USD" if a in USD else "CNY"):
            raise ValueError(f"{a}: currency specification mismatch")
        if float(spec.get("point_value", -1)) != (1 if a in FUNDED else POINTS[a]):
            raise ValueError(f"{a}: verify native numeric quote points and point value")
        if not spec.get("source") or not spec.get("price_type"):
            raise ValueError(f"{a}: source and price type are required")
        if a in FUNDED and spec.get("dividend_policy") != "excluded":
            raise ValueError("This original-underlying version declares price indices excluding dividends")
    raw = {}
    for name in FILES:
        raw[name] = (directory / name).read_bytes()
        expected = metadata.get("input_sha256", {}).get(name)
        if not isinstance(expected, str) or hashlib.sha256(raw[name]).hexdigest() != expected:
            raise ValueError(f"Input hash not approved or changed after gate: {name}")
    panels = {name: pd.read_csv(BytesIO(raw[name]), index_col=0, parse_dates=True)
              for name in FILES[1:]}
    market = pd.read_csv(BytesIO(raw[FILES[0]]), parse_dates=["date"])
    market["tradeable"] = _booleans(market[["tradeable"]], "tradeable").tradeable
    market = market.set_index(["date", "asset_id"]).sort_index()
    observed = _booleans(panels["native_observed.csv"], "native observed")
    common = _booleans(panels["common_execution_dates.csv"], "common calendar")
    if list(common.columns) != ["common_open"]:
        raise ValueError("Common calendar CSV needs one common_open column")
    data = ResearchInputs(market, panels["local_actual_leg_returns.csv"],
        panels["cny_actual_leg_risk_returns.csv"], observed, common.common_open, metadata)
    return validate_inputs(data)
