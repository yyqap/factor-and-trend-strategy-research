"""Training-window-only preprocessing and interpretable redundancy screening."""
import warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def tail_labels(panel, fraction=.30):
    """Label exactly floor(fraction*N) names in each tail; stock-code tie-break."""
    if not 0 < fraction < .5:
        raise ValueError("Tail fraction must lie between0and0.5")
    labels = pd.Series(np.nan, index=panel.index, dtype=float)
    for _, g in panel.groupby("signal_month", sort=True):
        ordered = g.sort_values(["target", "stock"], kind="stable")
        k = int(np.floor(fraction*len(g)))
        if k < 1:
            raise ValueError("Too few stocks to form both label tails")
        labels.loc[ordered.index[:k]] = 0.
        labels.loc[ordered.index[-k:]] = 1.
    return labels


def screen_features(train, features, threshold=.90):
    """Greedy correlation filter, prioritizing absolute TRAINING mean Rank IC.

    Monthly Spearman matrices are summarized by median absolute correlation.
    Retained columns preserve original input order. All decisions use `train`.
    """
    if not 0 < threshold <= 1:
        raise ValueError("Correlation threshold must be in(0,1]")
    usable = [f for f in features if train[f].nunique(dropna=True) > 1]
    if not usable:
        raise ValueError("No nonconstant observed factors in this training window")
    matrices, ics = [], []
    for _, g in train.groupby("signal_month", sort=True):
        x = g[usable]
        with np.errstate(invalid="ignore", divide="ignore"):
            matrices.append(x.corr(method="spearman").abs().to_numpy())
            ics.append(x.rank().corrwith(g.target.rank()).to_numpy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        matrix = np.nanmedian(np.stack(matrices), axis=0)
        mean_ic = np.nanmean(np.stack(ics), axis=0)
    priority = sorted(range(len(usable)), key=lambda j: (
        -abs(mean_ic[j]) if np.isfinite(mean_ic[j]) else 0., usable[j]))
    keep, audit = [], []
    for j in priority:
        matches = [k for k in keep if matrix[j, k] >= threshold]
        representative = max(matches, key=lambda k: matrix[j, k]) if matches else None
        audit.append({"feature": usable[j], "retained": not matches,
                      "training_mean_rank_ic": float(mean_ic[j]),
                      "representative": usable[representative] if representative is not None else None})
        if not matches:
            keep.append(j)
    audit.extend({"feature": f, "retained": False, "reason": "constant_or_all_missing_in_training"}
                 for f in features if f not in usable)
    return [usable[j] for j in sorted(keep)], audit


class TrainPreprocessor:
    """Fitted on all eligible TRAINING rows, including classifier middle40%."""
    def fit(self, train, features):
        self.features = list(features)
        self.last_training_signal = train.signal_month.max()
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()
        self.scaler.fit(self.imputer.fit_transform(train[self.features]))
        return self

    def transform(self, panel):
        return self.scaler.transform(self.imputer.transform(panel[self.features]))


def cross_sectional_mad_clip(panel, features, k=3.):
    """Optional RAW-factor utility; not applied to the supplied preprocessed panel.

    Clip to median +/- k*MAD within each signal month. MAD is unscaled median
    absolute deviation; a zero MAD leaves that column unchanged.
    """
    result = panel.copy()
    for _, g in result.groupby("signal_month"):
        values = g[features]
        center = values.median()
        mad = values.sub(center).abs().median()
        low, high = center-k*mad, center+k*mad
        valid = mad.gt(0)
        result.loc[g.index, np.array(features)[valid]] = values.loc[:, valid].clip(
            lower=low[valid], upper=high[valid], axis=1)
    return result
