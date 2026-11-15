"""Evaluation-only industry controls and return-factor attribution helpers."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata


def asof_industries(predictions, history):
    """Use latest PUBLIC snapshot at signal month-end; no later-label backfill.

    history columns: stock, industry, snapshot_date, published_date. Security
    keys must already match predictions; the adapter does not infer exchanges.
    Publication dates must be independently verified by the data owner.
    """
    required = {"stock", "industry", "snapshot_date", "published_date"}
    if not required.issubset(history):
        raise ValueError("Industry history requires security, classification and availability dates")
    h = history.copy()
    for col in ["snapshot_date", "published_date"]:
        h[col] = pd.to_datetime(h[col])
    if h.duplicated(["stock", "snapshot_date", "published_date"]).any() or h.snapshot_date.gt(h.published_date).any():
        raise ValueError("Ambiguous industry snapshot or invalid publication chronology")
    snapshots = h[["snapshot_date", "published_date"]].drop_duplicates().sort_values(["published_date", "snapshot_date"])
    joined = []
    for month, g in predictions.groupby("signal_month", sort=True):
        eligible = snapshots[snapshots.published_date.le(pd.Period(month, "M").end_time)]
        if eligible.empty:
            z = g.copy()
            z["industry"] = np.nan
            z["industry_published_date"] = pd.NaT
            z["industry_snapshot_date"] = pd.NaT
        else:
            chosen = eligible.iloc[-1]
            members = h[h.published_date.eq(chosen.published_date) & h.snapshot_date.eq(chosen.snapshot_date)]
            z = g.merge(members[["stock", "industry"]], how="left", on="stock", validate="many_to_one")
            z["industry_published_date"] = chosen.published_date
            z["industry_snapshot_date"] = chosen.snapshot_date
        joined.append(z)
    return pd.concat(joined, ignore_index=True)


def industry_rank_ic(frame, minimum_group=5):
    """Raw matched, two-sided raw residual IC, rank-first partial IC, Alphalens style.

    This uses realized future returns ONLY for ex-post evaluation. It does not
    create new forecasts, executable industry-neutral weights or pure alpha.
    """
    if minimum_group < 2:
        raise ValueError("Singleton groups contain no within-industry information")
    rows = []
    for (model, month), g in frame.groupby(["model", "signal_month"], sort=True):
        h = g[g.industry.notna() & g.industry.ne("UNKNOWN")].copy()
        h = h[h.industry.map(h.industry.value_counts()).ge(minimum_group)]
        if len(h) < 3:
            raise ValueError("Insufficient known, non-singleton industry coverage")
        codes = pd.Categorical(h.industry).codes
        def residual(a):
            return a-(np.bincount(codes, weights=a)/np.bincount(codes))[codes]
        def corr(a, b):
            return float(np.corrcoef(a, b)[0, 1])
        def rankic(a, b):
            return corr(rankdata(a), rankdata(b))
        score, target = h.score.to_numpy(), h.target.to_numpy()
        within, counts = [], []
        for _, members in h.groupby("industry", sort=True):
            value = rankic(members.score.to_numpy(), members.target.to_numpy())
            if np.isfinite(value):
                within.append(value)
                counts.append(len(members))
        raw, adjusted = rankic(score, target), rankic(residual(score), residual(target))
        rows.append({"model": model, "signal_month": month,
                     "label_month": str(pd.Period(month, "M")+1),
                     "minimum_group_stocks": minimum_group, "total_original_stocks": len(g),
                     "included_stocks": len(h),
                     "included_fraction": len(h)/len(g), "industry_count": h.industry.nunique(),
                     "full_original_rank_ic": rankic(g.score.to_numpy(), g.target.to_numpy()),
                     "raw_rank_ic_same_sample": raw,
                     "industry_adjusted_rank_ic": adjusted,
                     "industry_partial_spearman": corr(residual(rankdata(score)), residual(rankdata(target))),
                     "return_only_group_adjusted_ic": rankic(score, residual(target)),
                     "within_industry_n_weighted_ic": float(np.average(within, weights=counts)) if within else np.nan,
                     "industry_adjusted_minus_raw": adjusted-raw})
    return pd.DataFrame(rows)


def industry_time_inference(monthly, bootstrap_draws=10000, seed=20261114):
    """Historical industry-IC uncertainty: equal months, HAC6 and moving blocks.

    Accept concatenated minimum-group2/5/10 outputs. Bootstrap applies only to
    the primary5-stock sample and uses noncircular blocks3/6/12; all metrics
    reuse the same sampled month indices for paired comparisons. This does
    not adjust for all earlier model/research searches.
    """
    metrics = ["raw_rank_ic_same_sample", "industry_adjusted_rank_ic", "industry_partial_spearman",
               "return_only_group_adjusted_ic", "within_industry_n_weighted_ic", "industry_adjusted_minus_raw"]
    inference, bootstrap = [], []
    for (model, minimum), group in monthly.groupby(["model", "minimum_group_stocks"], sort=True):
        group = group.sort_values("signal_month")
        periods = pd.PeriodIndex(group.signal_month, freq="M")
        if group.signal_month.duplicated().any() or len(group) < 2 or not np.all(np.diff(periods.asi8) == 1):
            raise ValueError("IC inference requires unique contiguous monthly observations")
        n = len(group)
        for metric in metrics:
            a = group[metric].to_numpy(float)
            if not np.isfinite(a).all():
                raise ValueError("Undefined monthly industry IC cannot silently become zero")
            mean = a.mean()
            u = a-mean
            meat = float(u@u)
            for lag in range(1, min(6, n-1)+1):
                meat += 2*(1-lag/7)*float(u[lag:]@u[:-lag])
            se = np.sqrt(max(0., meat/(n*(n-1))))
            inference.append({"model": model, "minimum_group_stocks": minimum, "metric": metric,
                "mean": mean, "months": n, "hac_lag": 6, "hac_se": se,
                "hac_t": mean/se if se > 0 else np.nan,
                "hac_ci95_low": mean-1.959963984540054*se,
                "hac_ci95_high": mean+1.959963984540054*se,
                "positive_months": int(np.sum(a > 0)), "positive_month_fraction": np.mean(a > 0)})
            if minimum == 5:
                for block in [6, 3, 12]:
                    if n < block:
                        raise ValueError("Too few months for the prespecified12-month block sensitivity")
                    rng = np.random.default_rng(seed)
                    starts = rng.integers(0, n-block+1, size=(bootstrap_draws, int(np.ceil(n/block))))
                    indices = (starts[:, :, None]+np.arange(block)).reshape(bootstrap_draws, -1)[:, :n]
                    low, high = np.quantile(a[indices].mean(axis=1), [.025, .975])
                    bootstrap.append({"model": model, "metric": metric, "block_length": block,
                        "draws": bootstrap_draws, "mean": mean, "ci95_low": low, "ci95_high": high})
    return pd.DataFrame(inference), pd.DataFrame(bootstrap)


def factor_attribution(portfolio, factors, factor_columns, hac_lag=6):
    """Matched-date OLS with intercept and Bartlett HAC; alpha=12*intercept.

    Factor columns must be traded return series, NOT stock characteristics.
    The market factor must already be market-minus-RF. Do not subtract RF twice.
    Daily long/short factor aggregation conventions belong to the input owner.
    """
    keys = ["signal_month", "entry_date", "exit_date"]
    p, f = portfolio.copy(), factors.copy()
    for obj in [p, f]:
        for col in ["entry_date", "exit_date"]:
            obj[col] = pd.to_datetime(obj[col])
        if obj.duplicated(keys).any():
            raise ValueError("Attribution expects one portfolio and one factor row per holding period")
    joined = p.merge(f[keys+list(factor_columns)], on=keys, how="left", validate="one_to_one")
    if joined[factor_columns].isna().any().any():
        raise ValueError("Factor returns do not match every portfolio holding date")
    joined = joined.sort_values("entry_date")
    periods = pd.PeriodIndex(joined.signal_month, freq="M")
    if len(periods) > 1 and not np.all(np.diff(periods.asi8) == 1):
        raise ValueError("This HAC helper requires contiguous monthly observations")
    y = (joined.net_return-joined.rf_return).to_numpy()
    X = np.column_stack([np.ones(len(joined)), joined[factor_columns].to_numpy()])
    n, k = X.shape
    if n <= k or np.linalg.matrix_rank(X) != k or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Attribution requires a finite, full-rank design with residual degrees of freedom")
    coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
    residuals = y-X@coefficients
    scores = X*residuals[:, None]
    meat = scores.T@scores
    for lag in range(1, min(hac_lag, n-1)+1):
        gamma = scores[lag:].T@scores[:-lag]
        meat += (1-lag/(hac_lag+1))*(gamma+gamma.T)
    inverse = np.linalg.inv(X.T@X)
    covariance = inverse@meat@inverse*n/(n-k)
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    return {"months": n, "factor_columns": list(factor_columns), "coefficients": coefficients.tolist(),
            "hac_standard_errors": se.tolist(), "hac_lag": hac_lag,
            "annual_arithmetic_alpha": float(12*coefficients[0]),
            "annual_alpha_ci95_normal": (12*(coefficients[0]+np.array([-1, 1])*1.959963984540054*se[0])).tolist(),
            "residual_mean": float(residuals.mean())}
