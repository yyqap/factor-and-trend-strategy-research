"""Reconcile public aggregate files without access to the private source data.

These checks verify exported arithmetic and cross-file consistency. They do
not validate the omitted stock-level inputs or turn historical research into
an independently reproduced live record.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"


def test_historical_correspondence_records_bind_to_current_public_code():
    for task in ["task1", "task2"]:
        record_file = RESULTS / task / "portable_code_verification.json"
        record = json.loads(record_file.read_text())
        assert record["status"] == "PASS"
        hashes = record["verified_portable_code_sha256"]
        assert hashes
        for relative_path, expected in hashes.items():
            source = RESULTS.parent / relative_path
            assert source.is_file()
            assert hashlib.sha256(source.read_bytes()).hexdigest() == expected


def test_task1_catalog_identity_and_attribution_fee_scope():
    catalog = pd.read_csv(RESULTS / "task1/portfolio_metrics.csv")
    attribution = pd.read_csv(RESULTS / "task1/factor_attribution.csv")
    assert len(catalog) == 192
    assert not catalog.duplicated(["config_id", "cost_bps"]).any()
    assert catalog.config_id.nunique() == len(attribution) == 48
    assert not attribution.config_id.duplicated().any()
    assert catalog.groupby("config_id").cost_bps.apply(set).eq({0, 5, 10, 20}).all()
    assert attribution.attribution_cost_bps.eq(10).all()
    assert catalog.loc[catalog.cost_bps.ne(10), "ff5_alpha_monthly"].isna().all()
    joined = catalog[catalog.cost_bps.eq(10)].merge(attribution, on="config_id", validate="one_to_one")
    for family in ["ff3", "ff5"]:
        field = family + "_alpha_annualized_arithmetic"
        np.testing.assert_allclose(joined[field + "_x"], joined[field + "_y"], atol=1e-12)
        np.testing.assert_allclose(attribution[field], 12 * attribution[family + "_alpha_monthly"], atol=1e-12)


def test_task1_baseline_returns_nav_costs_and_catalog_metrics():
    returns = pd.read_csv(RESULTS / "task1/monthly_baseline_returns.csv")
    navs = pd.read_csv(RESULTS / "task1/monthly_baseline_nav.csv")
    catalog = pd.read_csv(RESULTS / "task1/portfolio_metrics.csv")
    for (model, fee), group in returns.groupby(["model", "cost_bps"]):
        g = group.sort_values("signal_month")
        n = navs[navs.model.eq(model) & navs.cost_bps.eq(fee)].sort_values("date")
        assert len(g) == 49 and len(n) == 50
        np.testing.assert_allclose(n.nav, np.r_[1.0, np.cumprod(1 + g.net_return)], atol=2e-12)
        dd = 1 - n.nav.to_numpy() / np.maximum.accumulate(n.nav.to_numpy())
        np.testing.assert_allclose(n.drawdown, dd, atol=2e-12)
        metric = catalog[catalog.model.eq(model) & catalog.cost_bps.eq(fee)
                         & catalog.policy_id.eq("top30_equal")].iloc[0]
        elapsed = (pd.Timestamp(g.exit_date.iloc[-1]) - pd.Timestamp(g.entry_date.iloc[0])).days
        assert metric.final_nav == pytest.approx(n.nav.iloc[-1])
        assert metric.cagr == pytest.approx(n.nav.iloc[-1] ** (365.25 / elapsed) - 1)
        assert metric.monthly_observation_nav_mdd == pytest.approx(dd.max())
        ir = np.sqrt(12) * g.active_return.mean() / g.active_return.std(ddof=1)
        assert metric.information_ratio == pytest.approx(ir)
        assert metric.mean_two_sided_target_turnover == pytest.approx(g.target_turnover_two_sided.mean())
        zero = returns[returns.model.eq(model) & returns.cost_bps.eq(0)].sort_values("signal_month")
        np.testing.assert_allclose(zero.net_return.to_numpy() - g.net_return.to_numpy(),
                                   fee / 10000 * g.target_turnover_two_sided.to_numpy(), atol=2e-12)


def test_task1_model_selection_and_industry_results_reconcile():
    models = pd.read_csv(RESULTS / "task1/model_comparison.csv").set_index("model")
    cv = pd.read_csv(RESULTS / "task1/cv_leaderboard.csv")
    industry = pd.read_csv(RESULTS / "task1/industry_monthly_ic.csv")
    summary = json.loads((RESULTS / "task1/summary.json").read_text())
    assert cv.loc[cv.mean_monthly_rank_ic.idxmax(), "model"] == summary["family_selected_by_development"]
    assert len(cv) == 18 and len(industry) == 147
    for model, g in industry.groupby("model"):
        assert len(g) == 49 and not g.signal_month.duplicated().any()
        assert models.loc[model, "cv_mean_rank_ic"] == pytest.approx(cv[cv.model.eq(model)].mean_monthly_rank_ic.max())
        assert models.loc[model, "industry_raw_matched_rank_ic"] == pytest.approx(g.raw_rank_ic_same_sample.mean())
        assert models.loc[model, "industry_adjusted_rank_ic"] == pytest.approx(g.industry_adjusted_rank_ic.mean())
        np.testing.assert_allclose(g.industry_adjusted_minus_raw,
                                   g.industry_adjusted_rank_ic - g.raw_rank_ic_same_sample, atol=2e-12)


def test_task2_nav_metrics_and_calendar_summaries():
    nav = pd.read_csv(RESULTS / "task2/strategy_nav.csv", parse_dates=["date"]).sort_values("date")
    metrics = pd.read_csv(RESULTS / "task2/performance.csv").set_index("window")
    assert len(nav) == 837
    r = nav.iloc[1:].net_return
    np.testing.assert_allclose(nav.nav, np.r_[1.0, np.cumprod(1 + r)], atol=2e-12)
    dd = 1 - nav.nav.to_numpy() / np.maximum.accumulate(nav.nav.to_numpy())
    np.testing.assert_allclose(nav.drawdown, dd, atol=2e-12)
    m = metrics.loc["full_native_window"]
    years = (nav.date.iloc[-1] - nav.date.iloc[0]).days / 365.25
    assert m.cagr == pytest.approx(nav.nav.iloc[-1] ** (1 / years) - 1)
    assert m.maximum_drawdown == pytest.approx(dd.max())
    assert m.annualized_volatility == pytest.approx(r.std(ddof=1) * np.sqrt(252))
    assert m.sharpe_rf_zero == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    actual = nav.iloc[1:].copy()
    for filename, key, labels in [("monthly_returns.csv", "month", actual.date.dt.to_period("M").astype(str)),
                                  ("yearly_returns.csv", "year", actual.date.dt.year)]:
        exported = pd.read_csv(RESULTS / "task2" / filename).set_index(key)
        calculated = actual.groupby(labels).net_return.apply(lambda x: (1 + x).prod() - 1)
        for label, value in calculated.items():
            assert exported.loc[label, "net_return"] == pytest.approx(value)
