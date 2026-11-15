"""Chronological model selection followed by frozen-parameter annual refits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import validate_panel
from .features import TrainPreprocessor, screen_features, tail_labels
from .models import fit_model, model_grid, predict_score


def default_config():
    return {"development_start": "2005-01", "development_end": "2014-05",
            "test_start": "2014-06", "test_end": "2018-06", "training_gap_months": 2,
            "folds": [{"validation_start": "2008-11", "validation_end": "2010-08"},
                      {"validation_start": "2010-09", "validation_end": "2012-06"},
                      {"validation_start": "2012-07", "validation_end": "2014-04"}],
            "correlation_threshold": .90, "tail_fraction": .30,
            "ridge_lambdas": [.0001, .001, .01, .1, 1., 10.],
            "logistic_C": [.01, .1, 1., 10.],
            "lightgbm_grid": {"num_leaves": [7, 15], "min_child_samples": [200, 1000],
                              "n_estimators": [100, 300]},
            "models": ["ridge", "logistic", "lightgbm"],
            "portfolio_sizes": [30, 50, 100], "weighting_rules": ["equal", "risk_rank"],
            "risk_feature": "std_3m", "cost_bps": [0, 5, 10, 20]}


def load_config(path=None):
    config = default_config()
    if path is not None:
        config.update(json.loads(Path(path).read_text()))
    return config


def training_rows(panel, prediction_month, config):
    cutoff = str(pd.Period(prediction_month, "M") - config["training_gap_months"])
    return panel.loc[panel.signal_month.between(config["development_start"], cutoff)].copy()


def monthly_rank_ic(frame, score):
    temp = frame[["signal_month", "target"]].copy()
    temp["score"] = np.asarray(score)
    rows = []
    for month, g in temp.groupby("signal_month", sort=True):
        value = g.score.rank(method="average").corr(g.target.rank(method="average"))
        if not np.isfinite(value):
            raise ValueError(f"Undefined full-universe Rank IC in {month}")
        rows.append({"signal_month": month, "rank_ic": value})
    return pd.DataFrame(rows)


def _prepare(train, validation, features, config):
    if train.empty or validation.empty:
        raise ValueError("A configured training/validation window is empty")
    selected, audit = screen_features(train, features, config["correlation_threshold"])
    preprocessor = TrainPreprocessor().fit(train, selected)
    return (selected, audit, preprocessor, preprocessor.transform(train),
            preprocessor.transform(validation), tail_labels(train, config["tail_fraction"]).to_numpy())


def _month_range(first, last):
    return list(pd.period_range(first, last, freq="M").astype(str))


def run_research(panel, features, config=None, models=None):
    """Return derived frames and protocol; no local files or external data are read.

    CV uses one fit per22-month block. Final evaluation uses annual expanding
    refits with the selected hyperparameters unchanged. These different refit
    schedules are disclosed rather than described as identical validation.
    """
    config = default_config() if config is None else config.copy()
    panel, features = validate_panel(panel, features)
    families = list(models or config["models"])
    if not families or len(set(families)) != len(families):
        raise ValueError("Choose one or more distinct model families")
    if config["training_gap_months"] < 2:
        raise ValueError("This target protocol requires a gap of at least2signal months")
    if pd.Period(config["development_end"], "M") >= pd.Period(config["test_start"], "M"):
        raise ValueError("Development and evaluation months overlap")
    expected = set(_month_range(config["development_start"], config["test_end"]))
    if not expected.issubset(set(panel.signal_month)):
        raise ValueError("Input is missing months required by the configured chronology")
    grids = {m: model_grid(m, config) for m in families}
    cv_rows, screen_rows, fit_logs = [], [], []
    for fold_id, fold in enumerate(config["folds"], 1):
        first, last = fold["validation_start"], fold["validation_end"]
        if pd.Period(last, "M") > pd.Period(config["test_start"], "M") - config["training_gap_months"]:
            raise ValueError("Validation outcomes are unavailable at the first test prediction")
        train = training_rows(panel, first, config)
        valid = panel.loc[panel.signal_month.between(first, last)].copy()
        selected, audit, prep, x, v, labels = _prepare(train, valid, features, config)
        screen_rows.extend({"fit": f"cv{fold_id}", **a} for a in audit)
        for family in families:
            for parameters in grids[family]:
                model = fit_model(family, parameters, x, train.target.to_numpy(), labels)
                scores = predict_score(model, family, v)
                ics = monthly_rank_ic(valid, scores)
                cv_rows.append({"model": family, "parameters": json.dumps(parameters, sort_keys=True),
                                "fold": fold_id, "validation_start": first, "validation_end": last,
                                "last_training_signal": prep.last_training_signal,
                                "validation_months": len(ics), "mean_rank_ic": float(ics.rank_ic.mean()),
                                "n_features": len(selected)})
    cv = pd.DataFrame(cv_rows)
    candidates = []
    for (family, parameters), g in cv.groupby(["model", "parameters"], sort=True):
        candidates.append({"model": family, "parameters": parameters,
                           "mean_validation_rank_ic": float(np.average(g.mean_rank_ic, weights=g.validation_months))})
    leaderboard = pd.DataFrame(candidates).sort_values(
        ["mean_validation_rank_ic", "model", "parameters"], ascending=[False, True, True])
    winners = {family: json.loads(leaderboard[leaderboard.model.eq(family)].iloc[0].parameters)
               for family in families}
    selection = {"criterion": "equal-month full-universe mean validation Rank IC",
                 "family_winner": leaderboard.iloc[0].model, "parameters": winners,
                 "selected_before_test_evaluation": True}
    predictions, fitted, year = [], {}, None
    for month in _month_range(config["test_start"], config["test_end"]):
        valid = panel.loc[panel.signal_month.eq(month)].copy()
        if not fitted or month[:4] != year:
            train = training_rows(panel, month, config)
            selected, audit, prep, x, _, labels = _prepare(train, valid, features, config)
            screen_rows.extend({"fit": "annual_" + month, **a} for a in audit)
            fitted = {family: fit_model(family, winners[family], x, train.target.to_numpy(), labels)
                      for family in families}
            year = month[:4]
            for family in families:
                fit_logs.append({"model": family, "first_prediction": month,
                                 "last_training_signal": prep.last_training_signal,
                                 "n_training_rows": len(train), "selected_features": selected,
                                 "parameters": winners[family]})
        v = prep.transform(valid)
        for family, model in fitted.items():
            result = valid[["stock", "signal_month", "target"]].copy()
            result["model"] = family
            result["score"] = predict_score(model, family, v)
            result["last_training_signal"] = prep.last_training_signal
            result["label_month"] = str(pd.Period(month, "M") + 1)
            if config["risk_feature"] in valid:
                result[config["risk_feature"]] = valid[config["risk_feature"]].to_numpy()
            predictions.append(result)
    predictions = pd.concat(predictions, ignore_index=True)
    return {"predictions": predictions, "cv_folds": cv, "leaderboard": leaderboard,
            "feature_screening": pd.DataFrame(screen_rows), "fit_logs": fit_logs,
            "selection": selection, "config": config}
