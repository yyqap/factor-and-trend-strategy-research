"""Synthetic analytical fixtures for the16-logical/20-leg wrapper only."""

from dataclasses import replace
import unittest
import numpy as np
import pandas as pd

from factor_trend_research.task2.native_engine import Instrument
from factor_trend_research.task2.native_strategy import DEFAULT_CONFIG, SignalFrames, run_native_strategy


def fixture(cost=0, ready_index=259, periods=360, basket_selected=False):
    # Weekdays here are an artificial known calendar, not actual exchange days.
    dates = pd.bdate_range("2020-01-01", periods=periods)
    logical = [f"L{i}" for i in range(15)] + ["BLACK"]
    actual = [f"A{i}" for i in range(15)] + [f"B{i}" for i in range(5)]
    instruments = [Instrument(a, "funded" if i < 5 else "future", "CNY", 1, cost)
                   for i, a in enumerate(actual)]
    mapping = {f"L{i}": {f"A{i}": 1.0} for i in range(15)}
    mapping["BLACK"] = {f"B{i}": .2 for i in range(5)}
    levels = pd.DataFrame(100., index=dates, columns=actual)
    risk = pd.DataFrame(0., index=dates, columns=logical)
    scores = pd.DataFrame(np.tile(np.linspace(1, .1, 16), (periods, 1)), index=dates, columns=logical)
    if basket_selected:
        scores["BLACK"] = 2
    scores.iloc[:ready_index] = np.nan
    vol = {n: pd.DataFrame(.2, index=dates, columns=logical) for n in (21,22,65,130,252)}
    corr = pd.DataFrame(0., index=dates, columns=logical)
    reversal = pd.DataFrame(False, index=dates, columns=logical)
    return {"dates": dates, "instruments": instruments, "mapping": mapping,
            "levels": levels, "risk": risk, "frames": SignalFrames(scores, vol, corr, reversal),
            "common": pd.Series(True, index=dates), "caps": dict.fromkeys(logical, .2),
            "cfg": replace(DEFAULT_CONFIG, transaction_cost=cost, management_fee=0),
            "start": dates[260], "end": dates[-1]}


def market_table(f):
    rows = []
    values, dates = f["levels"], f["dates"]
    kinds = {s.asset_id:s.kind for s in f["instruments"]}
    for i, d in enumerate(dates):
        for a in values.columns:
            rows.append({"date":d, "asset_id":a, "start_price":values[a].iloc[max(i-1,0)],
                         "end_price":values[a].iloc[i], "start_fx":1., "end_fx":1.,
                         "held_contract":"fixture-v1" if kinds[a]=="future" else None,
                         "next_contract":None, "next_price":None, "tradeable":True})
    return pd.DataFrame(rows).set_index(["date","asset_id"])


def run_fixture(f, **overrides):
    kwargs = {"market_long":market_table(f), "risk_returns":f["risk"], "common_open":f["common"],
              "caps":f["caps"], "instruments":f["instruments"], "leg_mapping":f["mapping"],
              "start":f["start"], "end":f["end"], "signal_frames":f["frames"], "cfg":f["cfg"],
              "analytical_fixture":True}
    kwargs.update(overrides)
    return run_native_strategy(**kwargs)


class NativeStrategyFixtures(unittest.TestCase):
    def test_full_four_day_phase_has_quarter_half_threequarter_full_exposure(self):
        result = run_fixture(fixture())
        phase = result.daily.index[result.daily.scheduled_rebalance][:4]
        self.assertEqual(len(phase),4)
        # .08/11 divided by20% annual vol; no risk scaling in a zero-return fixture.
        full_weight = .08/11/.2
        for fraction, d in zip((.25,.5,.75,1),phase):
            self.assertAlmostEqual(result.logical_weights.loc[d,"L0"], fraction*full_weight, places=10)
        self.assertEqual(result.daily.loc[phase,"phase_remaining"].tolist(),[3,2,1,0])
        starts=[x for x in result.decisions if x["event"]=="monthly_phase_start"]
        self.assertTrue(all(pd.Timestamp(x["signal_date"]) < pd.Timestamp(x["execution_date"]) for x in starts))

    def test_if_day4_unready_it_waits_until_next_month_not_day5(self):
        f=fixture()
        january=f["dates"][(f["dates"].year==2021)&(f["dates"].month==1)]
        day4=january[3]
        # The decision observed on day4 is unready, but day5 can see a ready row.
        f["frames"].score.loc[:day4-pd.offsets.BDay(1)]=np.nan
        result=run_fixture(f)
        first=result.daily.index[result.daily.scheduled_rebalance][0]
        self.assertEqual(first.month,2)
        january_run=result.daily.loc["2021-01"]
        self.assertFalse(january_run.scheduled_rebalance.any())
        self.assertTrue((january_run.nav==1_000_000).all())

    def test_new_position_earns_only_from_the_following_interval(self):
        f=fixture()
        baseline=run_fixture(f)
        entry=baseline.daily.index[baseline.daily.scheduled_rebalance][0]
        following=f["dates"][f["dates"].get_loc(entry)+1]
        f["levels"].loc[entry:,"A0"]=200
        f["levels"].loc[following:,"A0"]=202
        result=run_fixture(f)
        self.assertAlmostEqual(result.daily.loc[entry,"nav"],1_000_000,places=6)
        quantity=result.actual_positions.query("asset_id=='A0'").set_index("date").loc[entry,"quantity"]
        self.assertAlmostEqual(result.daily.loc[following,"funded_pnl_cny"], quantity*2,places=6)
        self.assertAlmostEqual(result.daily.loc[following,"nav"],1_000_000+quantity*2,places=6)

    def test_future_inputs_cannot_change_earlier_decisions_or_nav(self):
        f=fixture()
        before=run_fixture(f)
        cutoff=before.daily.index[35]
        later=f["dates"]>cutoff
        f["frames"].score.loc[later]=f["frames"].score.loc[later].iloc[:,::-1].to_numpy()
        for panel in f["frames"].vol.values():
            panel.loc[later]=.8
        f["risk"].loc[later]=.02
        f["levels"].loc[later,"A0"]=120
        after=run_fixture(f)
        pd.testing.assert_frame_equal(before.daily.loc[:cutoff],after.daily.loc[:cutoff],check_exact=True)
        pd.testing.assert_frame_equal(before.logical_weights.loc[:cutoff],after.logical_weights.loc[:cutoff],check_exact=True)
        previous=lambda r:[x for x in r.decisions if pd.Timestamp(x["execution_date"])<=cutoff]
        self.assertEqual(previous(before),previous(after))

    def test_daily_black_rebalance_preserves_group_notional_and_charges_two_sides(self):
        f=fixture(cost=.001,basket_selected=True)
        initial=run_fixture(f)
        phase=initial.daily.index[initial.daily.scheduled_rebalance][:4]
        shock=f["dates"][f["dates"].get_loc(phase[-1])+1]
        f["levels"].loc[shock:,"B0"]=200
        result=run_fixture(f)
        positions=result.actual_positions
        previous_date=f["dates"][f["dates"].get_loc(shock)-1]
        previous=positions.loc[positions.date==previous_date].set_index("asset_id")
        today=positions.loc[positions.date==shock].set_index("asset_id")
        old_leg=previous.loc["B0","futures_notional_cny"]
        total_pre_fee=6*old_leg
        for a in [f"B{i}" for i in range(5)]:
            self.assertAlmostEqual(today.loc[a,"futures_notional_cny"],total_pre_fee/5,places=6)
        self.assertAlmostEqual(result.daily.loc[shock,"traded_notional_cny"],1.6*old_leg,places=6)
        self.assertAlmostEqual(result.daily.loc[shock,"transaction_cost_cny"],.001*1.6*old_leg,places=6)
        self.assertEqual(today.loc["A0","quantity"],previous.loc["A0","quantity"])
        self.assertFalse(result.daily.loc[shock,"allocation_trade"])
        daily=result.daily
        residual=(daily.nav-daily.opening_nav_cny-daily.funded_pnl_cny-daily.futures_vm_cny
                  +daily.management_fee_cny+daily.transaction_cost_cny)
        self.assertLess(residual.abs().max(),1e-6)

    def test_unknown_risk_gap_and_unapproved_real_run_are_rejected(self):
        f=fixture()
        with self.assertRaisesRegex(ValueError,"passed data gate"):
            run_fixture(f,analytical_fixture=False)
        f["risk"].iloc[300,0]=np.nan
        with self.assertRaisesRegex(ValueError,"internal risk-return gaps"):
            run_fixture(f)

    def test_an_incomplete_four_session_window_does_not_start_a_phase(self):
        f=fixture()
        baseline=run_fixture(f)
        phase=baseline.daily.index[baseline.daily.scheduled_rebalance][:4]
        short=run_fixture(f,end=phase[1])
        self.assertFalse(short.daily.scheduled_rebalance.any())
        self.assertTrue((short.daily.nav==1_000_000).all())


if __name__=="__main__":
    unittest.main(verbosity=2)
