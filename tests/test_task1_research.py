import numpy as np
import pandas as pd
import pytest
from factor_trend_research.task1.data import load_panel, validate_panel
from factor_trend_research.task1.features import TrainPreprocessor, screen_features, tail_labels
from factor_trend_research.task1.research import default_config, run_research, training_rows


@pytest.fixture
def panel():
    rng = np.random.default_rng(24)
    rows = []
    for month in pd.period_range("2005-01", "2008-12", freq="M"):
        x = rng.normal(size=(24, 3))
        for i in range(24):
            rows.append({"stock": f"S{i:02}", "signal_month": str(month),
                         "target": .03*x[i, 0]+rng.normal(0, .04),
                         "x": x[i, 0], "duplicate_x": x[i, 0], "z": x[i, 1],
                         "std_3m": .1+abs(x[i, 2])})
    return pd.DataFrame(rows)


def small_config():
    c = default_config()
    c.update(development_end="2007-09", test_start="2007-10", test_end="2008-12",
             folds=[{"validation_start": "2005-11", "validation_end": "2006-04"},
                    {"validation_start": "2006-05", "validation_end": "2006-10"},
                    {"validation_start": "2006-11", "validation_end": "2007-04"}],
             ridge_lambdas=[.01, 1.], logistic_C=[.1, 1.], models=["ridge", "logistic"])
    return c


def test_reject_duplicate_month_stock_and_unrealized_target(panel):
    with pytest.raises(ValueError, match="Duplicate"):
        validate_panel(pd.concat([panel, panel.iloc[[0]]]))
    panel.loc[0, "target"] = np.nan
    with pytest.raises(ValueError, match="Targets"):
        validate_panel(panel)


def test_original_adapter_preserves_forward_target_without_extra_shift(tmp_path):
    for i, target in [(82, .11), (83, -.07)]:
        pd.DataFrame({"month": [i], "stock": ["000001.SZ"], "status": [1],
                      "return": [target], "EP": [i]}).to_csv(tmp_path/f"{i}.csv", index=False)
    out, factors, _ = load_panel(tmp_path, original_start=82, original_end=83)
    assert out.signal_month.tolist() == ["2005-01", "2005-02"]
    assert out.target.tolist() == [.11, -.07]
    assert factors == ["EP"]


def test_preprocessing_and_factor_choice_use_training_only(panel):
    config = small_config()
    train = training_rows(panel, "2007-10", config)
    assert train.signal_month.max() == "2007-08"
    selected, _ = screen_features(train, ["x", "duplicate_x", "z", "std_3m"])
    assert len(set(selected) & {"x", "duplicate_x"}) == 1
    pre = TrainPreprocessor().fit(train, selected)
    means = pre.scaler.mean_.copy()
    future = panel[panel.signal_month.ge("2007-10")].copy()
    future[selected] = 1e9
    pre.transform(future)
    assert np.array_equal(means, pre.scaler.mean_)


def test_balanced_tail_training_excludes_middle(panel):
    y = tail_labels(panel, .3)
    for _, indices in panel.groupby("signal_month").groups.items():
        assert y.loc[indices].eq(1).sum() == y.loc[indices].eq(0).sum() == 7
        assert y.loc[indices].isna().sum() == 10


def test_test_outcomes_do_not_select_hyperparameters_or_first_year_fit(panel):
    features = ["x", "duplicate_x", "z", "std_3m"]
    config = small_config()
    original = run_research(panel, features, config)
    changed = panel.copy()
    changed.loc[changed.signal_month.ge(config["test_start"]), "target"] *= -100
    updated = run_research(changed, features, config)
    assert original["selection"] == updated["selection"]
    p = original["predictions"]
    q = updated["predictions"]
    assert np.array_equal(p.loc[p.signal_month.lt("2008-01"), "score"],
                          q.loc[q.signal_month.lt("2008-01"), "score"])
    assert all(pd.Period(x["last_training_signal"], "M") <= pd.Period(x["first_prediction"], "M")-2
               for x in original["fit_logs"])
    assert len(p) == 2*15*24


def test_validation_cannot_touch_unavailable_test_outcomes(panel):
    config = small_config()
    config["folds"][-1]["validation_end"] = "2007-09"
    with pytest.raises(ValueError, match="unavailable"):
        run_research(panel, ["x", "z"], config)


def test_optional_lightgbm_is_real_requested_estimator(panel):
    pytest.importorskip("lightgbm")
    from factor_trend_research.task1.models import estimator
    from lightgbm import LGBMClassifier
    assert isinstance(estimator("lightgbm", {"num_leaves": 7, "min_child_samples": 10,
                                           "n_estimators": 5}, len(panel)), LGBMClassifier)
