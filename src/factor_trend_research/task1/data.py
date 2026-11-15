"""Adapters for user-owned panels. No proprietary dataset is distributed."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

KEYS = ["stock", "signal_month"]
METADATA = {"stock", "signal_month", "target", "month", "status", "return",
            "label_month", "model", "score", "last_training_signal"}


def file_record(path):
    path = Path(path)
    return {"filename": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def validate_panel(panel, features=None):
    """Reject ambiguous keys rather than silently duplicating or aggregating returns."""
    panel = panel.copy()
    if not {"stock", "signal_month", "target"}.issubset(panel):
        raise ValueError("Panel requires stock, signal_month, target columns")
    if panel.stock.isna().any() or panel.signal_month.isna().any():
        raise ValueError("Security identifiers and signal months cannot be missing")
    panel["stock"] = panel.stock.astype(str)
    panel["signal_month"] = panel.signal_month.map(lambda x: str(pd.Period(x, "M")))
    if panel.duplicated(KEYS).any():
        raise ValueError("Duplicate stock + signal_month keys")
    panel["target"] = pd.to_numeric(panel.target, errors="raise")
    if not np.isfinite(panel.target).all():
        raise ValueError("Targets must be observed finite forward relative returns")
    features = list(features) if features is not None else [c for c in panel if c not in METADATA]
    if not features or len(set(features)) != len(features):
        raise ValueError("Provide unique numeric factor columns")
    for col in features:
        if col in METADATA or col not in panel:
            raise ValueError(f"Invalid factor column: {col}")
        panel[col] = pd.to_numeric(panel[col], errors="raise").replace([np.inf, -np.inf], np.nan)
    return panel.sort_values(["signal_month", "stock"]).reset_index(drop=True), features


def load_panel(path, features=None, original_start=82, original_end=243):
    """Read normalized CSV or the original numbered monthly CSV layout.

    The numbered adapter uses month ID1 = 1998-04, status=1 eligibility,
    and excludes ID244 (unrealized zero placeholder labels). Its `return`
    column is already the next-holding-period target and is never shifted.
    """
    path = Path(path)
    records = []
    if path.is_dir():
        directory = path / "csv_demo" if (path / "csv_demo").is_dir() else path
        frames = []
        for number in range(original_start, original_end + 1):
            filename = directory / f"{number}.csv"
            raw = pd.read_csv(filename, dtype={"stock": str})
            if not {"month", "stock", "status", "return"}.issubset(raw):
                raise ValueError(f"Missing original columns in {filename.name}")
            if not raw.month.eq(number).all() or raw.duplicated(["stock", "month"]).any():
                raise ValueError(f"Ambiguous original month/security keys in {filename.name}")
            raw = raw.loc[raw.status.eq(1)].copy()
            raw["signal_month"] = str(pd.Period("1998-04", "M") + number - 1)
            raw["target"] = raw["return"]
            frames.append(raw)
            records.append(file_record(filename))
        panel = pd.concat(frames, ignore_index=True)
    else:
        panel = pd.read_csv(path, dtype={"stock": str})
        records.append(file_record(path))
    panel, selected = validate_panel(panel, features)
    quality = {"rows": len(panel), "stocks_historical_union": panel.stock.nunique(),
               "months": panel.signal_month.nunique(), "features": selected,
               "missing_factor_cells": int(panel[selected].isna().sum().sum()),
               "sources": records, "synthetic": False}
    return panel, selected, quality


def synthetic_panel(seed=20261114, n_stocks=128, n_features=70):
    """Deterministic implementation fixture; results are NOT market performance."""
    if n_features < 4 or n_stocks < 5:
        raise ValueError("Synthetic fixture needs at least4factors and5stocks")
    rng = np.random.default_rng(seed)
    features = [f"factor_{i:02d}" for i in range(n_features - 1)] + ["std_3m"]
    frames = []
    for month in pd.period_range("2005-01", "2018-06", freq="M"):
        x = rng.normal(size=(n_stocks, n_features))
        x[:, 2] = x[:, 1] + rng.normal(0, .015, n_stocks)
        x[:, -1] = rng.uniform(.05, .45, n_stocks)
        target = .012*x[:, 0] + .006*x[:, 1] + .006*x[:, 0]*x[:, 3]
        target += rng.normal(0, .05, n_stocks) + rng.normal(0, .01)
        missing = rng.random(x.shape) < .005
        missing[:, -1] = False  # Risk-weight fixture remains explicitly observable.
        x[missing] = np.nan
        g = pd.DataFrame(x, columns=features)
        g["stock"] = [f"S{i:04d}" for i in range(n_stocks)]
        g["signal_month"] = str(month)
        g["target"] = target
        frames.append(g)
    panel, features = validate_panel(pd.concat(frames, ignore_index=True), features)
    return panel, features, {"synthetic": True, "seed": seed, "rows": len(panel),
                             "months": 162, "features": features,
                             "notice": "SYNTHETIC SOFTWARE DEMONSTRATION; not real investment results"}
