"""Analytical fixtures only; no acquired market data or portfolio backtest."""

from datetime import datetime
import unittest

from factor_trend_research.task2.native_engine import (
    Instrument, IntervalMark, LastMark, PortfolioState,
    account_interval, expand_target_weights,
)


class NativeAccountingFixtures(unittest.TestCase):
    def assert_money(self, observed, expected):
        self.assertAlmostEqual(observed, expected, delta=1e-6)

    def test_fx_marks_real_funded_principal_but_not_futures_principal(self):
        specs = [Instrument("index", "funded", "USD", one_way_cost_rate=0),
                 Instrument("note", "future", "USD", 1000, 0)]
        state = PortfolioState(1_000_000, {"index": 100}, {"note": 1}, {"note": "202003"})
        marks = {"index": IntervalMark(100, 100, 7, 7.2),
                 "note": IntervalMark(100, 100, 7, 7.2, "202003")}
        result = account_interval(state, specs, marks,
                                  start_date="2020-02-24", end_date="2020-02-25")
        self.assert_money(result.opening_nav_cny, 1_070_000)
        self.assert_money(result.funded_pnl_cny["index"], 2_000)
        self.assert_money(result.futures_vm_cny["note"], 0)
        self.assert_money(result.end_nav_cny, 1_072_000)
        self.assert_money(result.state.cash_cny, 1_000_000)
        self.assert_money(result.futures_notional_cny["note"], 720_000)
        self.assertEqual(result.state.future_quantities["note"], 1)

    def test_same_contract_point_pnl_is_cash_and_not_funded_value(self):
        specs = [Instrument("note", "future", "USD", 1000, 0)]
        state = PortfolioState(1_000_000, future_quantities={"note": 2},
                               future_contracts={"note": "202003"})
        result = account_interval(state, specs,
                                  {"note": IntervalMark(100, 101, 7, 7.2, "202003")},
                                  start_date="2020-02-24", end_date="2020-02-25")
        self.assert_money(result.futures_vm_native["note"], 2_000)
        self.assert_money(result.futures_vm_cny["note"], 14_400)
        self.assert_money(result.end_nav_cny, 1_014_400)
        self.assertEqual(result.funded_market_value_cny, {})
        self.assertEqual(result.cash_weight, 1)
        self.assertEqual(result.state.future_quantities["note"], 2)

    def test_native_zt_point_value_not_face_value(self):
        specs = [Instrument("ZT", "future", "USD", 2000, 0)]
        state = PortfolioState(1_000_000, future_quantities={"ZT": 1},
                               future_contracts={"ZT": "202003"})
        # A one-eighth-of-one-thirty-second point move is USD7.8125.
        m = IntervalMark(107, 107 + 1 / 256, 7, 7, "202003")
        result = account_interval(state, specs, {"ZT": m},
                                  start_date="2020-02-24", end_date="2020-02-25")
        self.assert_money(result.futures_vm_native["ZT"], 7.8125)
        self.assert_money(result.futures_vm_cny["ZT"], 54.6875)

    def test_roll_two_legs_without_roll_gap_pnl_then_next_interval(self):
        specs = [Instrument("note", "future", "USD", 1000, 0.0005)]
        state = PortfolioState(1_000_000, future_quantities={"note": 1},
                               future_contracts={"note": "202003"})
        result = account_interval(
            state, specs,
            {"note": IntervalMark(100, 102, 7, 7, "202003", "202006", 110)},
            start_date="2020-02-26", end_date="2020-02-27")
        # P&L is the OLD contract's two-point move, not the 102 -> 110 gap.
        self.assert_money(result.futures_vm_cny["note"], 14_000)
        self.assert_money(result.transaction_cost_cny, 714)
        self.assert_money(result.gross_traded_notional_cny, 1_428_000)
        self.assert_money(result.end_nav_cny, 1_013_286)
        self.assertEqual([t.action for t in result.trades], ["roll_close", "roll_open"])
        self.assertEqual([t.contract for t in result.trades], ["202003", "202006"])
        self.assertAlmostEqual(result.state.future_quantities["note"], 102 / 110)
        self.assertEqual(result.state.last_marks["note"].price, 110)
        following = account_interval(
            result.state, specs,
            {"note": IntervalMark(110, 111, 7, 7, "202006")},
            start_date="2020-02-27", end_date="2020-02-28")
        self.assert_money(following.futures_vm_cny["note"], (102 / 110) * 7_000)
        self.assertEqual(following.transaction_cost_cny, 0)

    def test_roll_can_preserve_contract_count_by_explicit_choice(self):
        specs = [Instrument("note", "future", "USD", 1000, 0.0005)]
        state = PortfolioState(1_000_000, future_quantities={"note": 1},
                               future_contracts={"note": "202003"})
        result = account_interval(
            state, specs,
            {"note": IntervalMark(100, 102, 7, 7, "202003", "202006", 110)},
            start_date="2020-02-26", end_date="2020-02-27", roll_policy="quantity")
        self.assertEqual(result.state.future_quantities["note"], 1)
        self.assert_money(result.transaction_cost_cny, (102 + 110) * 1000 * 7 * 0.0005)

    def test_post_fee_targets_preserve_cash_plus_funded_nav(self):
        specs = [Instrument("index", "funded", "CNY", one_way_cost_rate=0.001),
                 Instrument("note", "future", "USD", 1000, 0.001)]
        state = PortfolioState(1_000_000)
        marks = {"index": IntervalMark(10, 10),
                 "note": IntervalMark(100, 100, 7, 7, "202003")}
        result = account_interval(state, specs, marks,
                                  start_date="2020-02-24", end_date="2020-02-25",
                                  target_weights={"index": 0.4, "note": 0.3})
        # NAV = 1m - .001 * (.4 + .3) * NAV.
        expected_nav = 1_000_000 / 1.0007
        self.assert_money(result.end_nav_cny, expected_nav)
        self.assert_money(result.transaction_cost_cny, 1_000_000 - expected_nav)
        self.assertAlmostEqual(result.funded_weights["index"], 0.4, places=10)
        self.assertAlmostEqual(result.futures_notional_weights["note"], 0.3, places=10)
        self.assertAlmostEqual(result.cash_weight, 0.6, places=10)
        self.assertAlmostEqual(result.cash_weight + sum(result.funded_weights.values()), 1)
        self.assert_money(result.state.cash_cny, expected_nav * 0.6)
        # Opening a futures position does NOT spend its 30% notional principal.
        self.assertGreater(result.state.cash_cny, expected_nav * 0.59)

    def test_funded_sale_returns_principal_but_futures_close_does_not(self):
        specs = [Instrument("index", "funded", "CNY", one_way_cost_rate=0.001),
                 Instrument("note", "future", "USD", 1000, 0.001)]
        state = PortfolioState(500_000, {"index": 50_000}, {"note": 1}, {"note": "202003"})
        result = account_interval(
            state, specs,
            {"index": IntervalMark(10, 10),
             "note": IntervalMark(100, 100, 7, 7, "202003")},
            start_date="2020-02-24", end_date="2020-02-25",
            target_weights={"index": 0, "note": 0})
        self.assert_money(result.transaction_cost_cny, 500 + 700)
        self.assert_money(result.state.cash_cny, 998_800)
        self.assert_money(result.end_nav_cny, 998_800)
        self.assertEqual(result.state.future_quantities["note"], 0)

    def test_closed_future_roll_is_deferred_without_fake_trade(self):
        specs = [Instrument("note", "future", "USD", 1000, 0.0005)]
        state = PortfolioState(1_000_000, future_quantities={"note": 1},
                               future_contracts={"note": "202003"})
        result = account_interval(
            state, specs,
            {"note": IntervalMark(100, 100, 7, 7, "202003", "202006", 110, False)},
            start_date="2020-02-24", end_date="2020-02-25")
        self.assertEqual(result.trades, ())
        self.assertEqual(result.transaction_cost_cny, 0)
        self.assertEqual(result.state.future_contracts["note"], "202003")
        self.assertEqual(result.state.future_quantities["note"], 1)
        self.assertEqual(result.deferred, ("note",))

    def test_frozen_funded_holdings_do_not_create_cash_to_buy_others(self):
        specs = [Instrument("A", "funded", "CNY", one_way_cost_rate=0),
                 Instrument("B", "funded", "CNY", one_way_cost_rate=0)]
        state = PortfolioState(100, {"A": 90})
        with self.assertRaisesRegex(ValueError, "unaffordable"):
            account_interval(
                state, specs,
                {"A": IntervalMark(10, 10, tradeable=False), "B": IntervalMark(10, 10)},
                start_date="2020-02-24", end_date="2020-02-25",
                target_weights={"A": 0.4, "B": 0.5})
        self.assertEqual(state.cash_cny, 100)
        self.assertEqual(state.funded_units, {"A": 90})

    def test_five_black_legs_preserve_logical_weight_and_residual_cash(self):
        legs = ["RB", "HC", "I", "J", "JM"]
        specs = [Instrument(a, "future", "CNY", 10) for a in legs]
        specs += [Instrument("CSI300", "funded", "CNY")]
        mapping = {"black": {a: 0.2 for a in legs}, "CSI300": {"CSI300": 1}}
        actual = expand_target_weights({"black": 0.15, "CSI300": 0.25}, mapping, specs)
        self.assertAlmostEqual(sum(actual.values()), 0.4)
        for a in legs:
            self.assertAlmostEqual(actual[a], 0.03)
        self.assertEqual(actual["CSI300"], 0.25)

    def test_explicit_management_fee_and_interval_reconciliation(self):
        specs = [Instrument("note", "future", "CNY", 10, 0)]
        state = PortfolioState(100_000, future_quantities={"note": 2},
                               future_contracts={"note": "202003"})
        result = account_interval(
            state, specs, {"note": IntervalMark(2300, 2303, held_contract="202003")},
            start_date="2020-02-24", end_date="2020-02-25", management_fee_cny=10)
        self.assert_money(result.end_nav_cny, 100_050)
        self.assert_money(result.pretrade_nav_cny, 100_050)
        self.assert_money(result.futures_vm_cny["note"], 60)

    def test_wrong_contract_or_missing_roll_overlap_rejected(self):
        specs = [Instrument("note", "future", "USD", 1000, 0)]
        state = PortfolioState(1_000_000, future_quantities={"note": 1},
                               future_contracts={"note": "202003"})
        for mark in (IntervalMark(100, 110, 7, 7, "202006"),
                     IntervalMark(100, 102, 7, 7, "202003", "202006")):
            with self.assertRaises(ValueError):
                account_interval(state, specs, {"note": mark},
                                 start_date="2020-02-24", end_date="2020-02-25")

    def test_price_and_fx_continuity_rejected_before_any_state_change(self):
        specs = [Instrument("note", "future", "USD", 1000, 0)]
        state = PortfolioState(1_000_000, future_quantities={"note": 1},
                               future_contracts={"note": "202003"},
                               last_marks={"note": LastMark(100, 7, "202003")},
                               as_of="2020-02-24")
        for mark in (IntervalMark(99, 101, 7, 7, "202003"),
                     IntervalMark(100, 101, 6.9, 7, "202003")):
            with self.assertRaisesRegex(ValueError, "prior end"):
                account_interval(state, specs, {"note": mark},
                                 start_date="2020-02-24", end_date="2020-02-25")
        self.assertEqual(state.last_marks["note"].price, 100)

    def test_unresolved_timestamp_and_short_targets_are_rejected(self):
        specs = [Instrument("index", "funded", "CNY")]
        marks = {"index": IntervalMark(10, 10)}
        for kwargs in (
                {"start_date": datetime(2020, 2, 24), "end_date": "2020-02-25"},
                {"start_date": "2020-02-24", "end_date": "2020-02-25",
                 "target_weights": {"index": -0.1}}):
            with self.assertRaises(ValueError):
                account_interval(PortfolioState(1000), specs, marks, **kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
