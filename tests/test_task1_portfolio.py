import numpy as np
import pandas as pd
import pytest
from factor_trend_research.task1.portfolio import backtest, performance_metrics, select_holdings, target_turnover
from factor_trend_research.task1.diagnostics import asof_industries, industry_rank_ic, industry_time_inference
from factor_trend_research.task1.regimes import market_exposure, relative_defensive_holdings


def predictions():
    return pd.DataFrame({"model": ["ridge"]*4, "signal_month": ["2014-06"]*2+["2014-07"]*2,
                         "stock": ["A", "B", "A", "B"], "score": [2., 1., 1., 2.],
                         "target": [.1, -.1, .9, -.2], "std_3m": [.3, .1, .2, .4]})


def test_stock_month_matching_and_initial_turnover():
    monthly, holdings, metrics = backtest(predictions(), n=1, cost_bps=10)
    assert np.allclose(monthly.gross_weighted_target, [.1, -.2])
    assert np.allclose(monthly.turnover, [1., 2.])
    assert np.allclose(monthly.cost, [.001, .002])
    assert holdings.groupby("signal_month").stock.nunique().eq(1).all()
    assert "sharpe" not in metrics and "cagr_actual_days" not in metrics
    assert target_turnover({"B": 1.}, {"A": 1.}) == 2.


def test_absolute_reconstruction_cash_leg_cost_and_nav_mdd():
    benchmark = pd.DataFrame({"signal_month": ["2014-06", "2014-07"],
        "entry_date": ["2014-07-01", "2014-08-01"], "exit_date": ["2014-08-01", "2014-09-01"],
        "benchmark_return": [.05, -.1], "rf_return": [.001, .001]})
    exposure = pd.Series([1., .5], index=benchmark.signal_month)
    monthly, _, metric = backtest(predictions(), n=1, cost_bps=10, benchmark=benchmark, exposures=exposure)
    assert np.allclose(monthly.net_return, [.149, -.151])
    assert np.allclose(monthly.turnover, [1., 1.5])
    assert metric.monthly_boundary_nav_mdd.iloc[0] == pytest.approx(.151)
    assert metric.final_nav.iloc[0] == pytest.approx(1.149*.849)


def test_initial_nav_is_in_drawdown_peak():
    m = pd.DataFrame({"signal_month": ["2020-01", "2020-02"], "net_relative": [-.2, -.1],
                      "turnover": [1., 0.], "cost": [0., 0.], "net_return": [-.2, -.1],
                      "rf_return": [0., 0.], "entry_date": ["2020-02-03", "2020-03-02"],
                      "exit_date": ["2020-03-02", "2020-04-01"]})
    assert performance_metrics(m)["monthly_boundary_nav_mdd"] == pytest.approx(.28)


def test_partial_exposure_without_benchmark_is_rejected():
    with pytest.raises(ValueError, match="Cash overlays"):
        backtest(predictions(), n=1, exposures=pd.Series({"2014-06": .5, "2014-07": .5}))


def test_risk_rank_uses_entire_universe_not_just_chosen_names():
    g = pd.DataFrame({"stock": ["A", "B", "C", "D"], "score": [4, 3, 2, 1],
                      "target": [0., 0., 0., 0.], "std_3m": [1., 2., 3., 4.]})
    h = select_holdings(g, n=2, weighting="risk_rank")
    assert np.allclose(h.weight, [4/7, 3/7])
    assert h.weight.sum() == pytest.approx(1.)


def test_industry_group_offsets_removed_and_future_labels_excluded():
    rng = np.random.default_rng(2)
    signal = np.r_[rng.normal(size=15), rng.normal(size=15)]
    group_offset = np.r_[np.repeat(-20., 15), np.repeat(20., 15)]
    p = pd.DataFrame({"model": "ridge", "signal_month": "2014-06",
                      "stock": [f"S{i}" for i in range(30)], "score": signal+group_offset,
                      "target": signal+2*group_offset})
    history = pd.DataFrame({"stock": p.stock, "industry": ["A"]*15+["B"]*15,
                            "snapshot_date": "2014-03-31", "published_date": "2014-04-03"})
    later = history.copy()
    later["industry"] = "FUTURE"
    later["snapshot_date"] = "2014-06-30"
    later["published_date"] = "2014-07-01"
    joined = asof_industries(p, pd.concat([history, later]))
    assert set(joined.industry) == {"A", "B"}
    result = industry_rank_ic(joined)
    assert result.industry_adjusted_rank_ic.iloc[0] == pytest.approx(1.)


def test_market_rule_never_uses_later_prices():
    dates = pd.bdate_range("2012-01-02", "2014-12-31")
    p = pd.DataFrame({"date": dates, "close": np.linspace(100., 200., len(dates))})
    base = market_exposure(p, ["2014-06"], "vol63_target15")
    p.loc[p.date.gt("2014-06-30"), "close"] *= 100
    changed = market_exposure(p, ["2014-06"], "vol63_target15")
    assert base.equals(changed)
    assert base.max_market_data_date.iloc[0] <= pd.Timestamp("2014-06-30")


def test_relative_state_ignores_unavailable_recent_targets():
    p = predictions()
    hist = pd.DataFrame({"signal_month": pd.period_range("2013-01", "2014-07", freq="M").astype(str),
                         "target": .01})
    _, states = relative_defensive_holdings(p, hist, lookback=6, candidate_pool=100, n=1)
    changed = hist.copy()
    changed.loc[changed.signal_month.ge("2014-06"), "target"] = -100
    _, changed_states = relative_defensive_holdings(p, changed, lookback=6, candidate_pool=100, n=1)
    assert states.equals(changed_states)


def test_defensive_equal_risk_ties_prefer_higher_score_then_stock():
    p = pd.DataFrame({"model": "ridge", "signal_month": "2014-06",
        "stock": [f"S{i:03d}" for i in range(100)], "score": np.arange(100),
        "target": 0., "std_3m": 1.})
    history = pd.DataFrame({"signal_month": pd.period_range("2013-01", "2014-06", freq="M").astype(str),
                            "target": .01})
    holdings, _ = relative_defensive_holdings(p, history, candidate_pool=100, n=30, always_defensive=True)
    assert set(holdings.stock) == {f"S{i:03d}" for i in range(70, 100)}


def test_relative_window_must_contain_consecutive_known_months():
    history = pd.DataFrame({"signal_month": pd.period_range("2013-01", "2014-07", freq="M").astype(str),
                            "target": .01})
    history = history[history.signal_month.ne("2014-03")]
    with pytest.raises(ValueError, match="Insufficient"):
        relative_defensive_holdings(predictions(), history, lookback=6, n=1)


def test_industry_paired_zero_difference_keeps_zero_uncertainty():
    rows = []
    for i, month in enumerate(pd.period_range("2010-01", periods=24, freq="M")):
        row = {"model": "ridge", "signal_month": str(month), "minimum_group_stocks": 5,
               "raw_rank_ic_same_sample": .1+.03*np.sin(i),
               "industry_adjusted_rank_ic": .1+.03*np.sin(i),
               "industry_partial_spearman": .1+.03*np.sin(i),
               "return_only_group_adjusted_ic": .1+.03*np.sin(i),
               "within_industry_n_weighted_ic": .1+.03*np.sin(i),
               "industry_adjusted_minus_raw": 0.}
        rows.append(row)
    inference, bootstrap = industry_time_inference(pd.DataFrame(rows), bootstrap_draws=100)
    paired = inference[inference.metric.eq("industry_adjusted_minus_raw")].iloc[0]
    assert paired.hac_ci95_low == paired.hac_ci95_high == paired.hac_se == 0.
    paired_boot = bootstrap[bootstrap.metric.eq("industry_adjusted_minus_raw")]
    assert paired_boot.ci95_low.eq(0).all() and paired_boot.ci95_high.eq(0).all()
