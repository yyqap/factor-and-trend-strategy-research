"""Analytical, hand-calculated accounting oracles; no historical backtest."""

import unittest

from factor_trend_research.task2.future_accounting import (
    funded_asset_pnl_cny,
    funded_asset_return_cny,
    funded_asset_value_cny,
    futures_notional_cny,
    futures_pnl_cny,
    futures_roll_cost_cny,
    futures_trade_cost_cny,
)


class FuturesAccountingTests(unittest.TestCase):
    def test_flat_future_fx_move_changes_notional_but_not_pnl(self):
        # One contract, $100 per point, fixed100 points; FX7.0 ->7.2.
        # Marked notional rises70,000 ->72,000CNY, but no price P&L exists.
        self.assertEqual(futures_pnl_cny(1, 100, 100, 100, 7.2), 0)
        self.assertEqual(futures_notional_cny(1, 100, 100, 7), 70000)
        self.assertEqual(futures_notional_cny(1, 100, 100, 7.2), 72000)
        # A funded holding of100 units at$100 really DOES have principal FX P&L.
        self.assertEqual(funded_asset_pnl_cny(100, 100, 100, 7, 7.2), 2000)

    def test_price_gain_is_translated_at_current_fx(self):
        # Three contracts * $1,000/point *2points = $6,000; at7.2 =43,200CNY.
        self.assertEqual(futures_pnl_cny(3, 1000, 100, 102, 7.2), 43200)
        self.assertEqual(futures_notional_cny(3, 1000, 102, 7.2), 2203200)

    def test_short_position_gains_when_price_falls(self):
        self.assertAlmostEqual(futures_pnl_cny(-2, 100, 80, 75, 7.1), 7100)

    def test_cny_contract_uses_unit_fx(self):
        self.assertEqual(futures_pnl_cny(2, 10, 2300, 2303, 1), 60)

    def test_constant_fx_notional_change_reconciles_to_price_pnl(self):
        before = futures_notional_cny(4, 1000, 100, 7)
        after = futures_notional_cny(4, 1000, 101, 7)
        self.assertEqual(after - before, 28000)
        self.assertEqual(futures_pnl_cny(4, 1000, 100, 101, 7), 28000)

    def test_daily_cny_cash_conversion_uses_each_days_fx(self):
        # $100 gain converted at7, then$200 at7.2 =700+1440=2140CNY.
        # This explicitly assumes each gain is converted, not retained as USD cash.
        first = futures_pnl_cny(1, 100, 100, 101, 7)
        second = futures_pnl_cny(1, 100, 101, 103, 7.2)
        self.assertEqual(first + second, 2140)

    def test_buy_and_sell_pay_same_one_way_notional_rate(self):
        # Each side trades3*$1,000*100*7.2=2,160,000CNY;5bp is1,080CNY.
        self.assertEqual(futures_trade_cost_cny(3, 1000, 100, 7.2, .0005), 1080)
        self.assertEqual(futures_trade_cost_cny(-3, 1000, 100, 7.2, .0005), 1080)
        self.assertEqual(futures_trade_cost_cny(0, 1000, 100, 7.2, .0005), 0)

    def test_roll_charges_both_legs_not_net_notional_difference(self):
        # Old exit1.4m + new entry1.47m =2.87m;5bp total fee1,435CNY.
        roll = futures_roll_cost_cny(2, 1000, 100, 7,
                                     2, 1000, 105, 7, .0005)
        self.assertEqual(roll.exit_notional_cny, 1400000)
        self.assertEqual(roll.entry_notional_cny, 1470000)
        self.assertEqual(roll.exit_cost_cny, 700)
        self.assertEqual(roll.entry_cost_cny, 735)
        self.assertEqual(roll.gross_traded_notional_cny, 2870000)
        self.assertEqual(roll.total_cost_cny, 1435)
        self.assertNotEqual(roll.total_cost_cny, 35)  # Incorrectly netting both legs.

    def test_roll_gap_is_not_profit_and_next_contract_marks_separately(self):
        # The old contract is flat at100. The new contract costs105 at entry.
        # The5-point delivery-price gap is not earned. Only105 ->106 earns14,000.
        old_pnl = futures_pnl_cny(2, 1000, 100, 100, 7)
        new_pnl = futures_pnl_cny(2, 1000, 105, 106, 7)
        self.assertEqual(old_pnl, 0)
        self.assertEqual(new_pnl, 14000)
        self.assertEqual(old_pnl + new_pnl, 14000)

    def test_roll_can_change_quantity_and_support_one_sided_entry_or_exit(self):
        # Different quantities are explicit; they are not inferred from contract prices.
        roll = futures_roll_cost_cny(1, 1000, 100, 7,
                                     2, 1000, 50, 7, .0005)
        self.assertEqual(roll.total_cost_cny, 700)
        entry = futures_roll_cost_cny(0, 1000, 100, 7,
                                      2, 1000, 50, 7, .0005)
        exit_only = futures_roll_cost_cny(1, 1000, 100, 7,
                                         0, 1000, 50, 7, .0005)
        self.assertEqual(entry.total_cost_cny, 350)
        self.assertEqual(exit_only.total_cost_cny, 350)

    def test_funded_price_and_fx_returns_compound(self):
        # Price+10%, FX+5% =>15.5%, including the cross term.
        self.assertAlmostEqual(funded_asset_return_cny(100, 110, 7, 7.35), .155)
        self.assertAlmostEqual(funded_asset_value_cny(100, 110, 7.35), 80850)
        self.assertAlmostEqual(funded_asset_pnl_cny(100, 100, 110, 7, 7.35), 10850)
        # Same numerical futures price movement has only translated price P&L.
        self.assertAlmostEqual(futures_pnl_cny(100, 1, 100, 110, 7.35), 7350)

    def test_invalid_inputs_do_not_silently_hide_bad_quotes(self):
        bad_calls = [
            lambda: futures_pnl_cny(1, 0, 100, 101, 7),
            lambda: futures_pnl_cny(1, 100, 100, 101, 0),
            lambda: futures_pnl_cny(0, 100, float("nan"), 101, 7),
            lambda: futures_notional_cny(float("inf"), 100, 100, 7),
            lambda: futures_trade_cost_cny(1, 100, 100, 7, -.0005),
            lambda: futures_trade_cost_cny(1, 100, 100, 7, float("nan")),
            lambda: funded_asset_return_cny(0, 110, 7, 7.35),
        ]
        for call in bad_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


if __name__ == "__main__":
    unittest.main(verbosity=2)
