"""Small accounting primitives; NOT integrated into a full strategy run.

Futures prices must already be parsed into numeric quote points. ``multiplier``
is native-currency P&L per one-point price change per contract. FX is CNY per
one unit of native currency; use 1 for CNY contracts. Quantities may be signed
and fractional for research. No contract specification is guessed here.

Futures P&L is translated at the supplied interval-end FX rate. Any actual USD
cash/margin balance, collateral interest, funding, margin calls, tax and price
slippage require separate accounting. Changing FX changes marked notional but
does not create a return on imaginary invested futures principal.

Trading costs here are ASSUMED one-way fractions of gross traded notional,
not a broker's per-contract commission schedule. All functions are standalone;
no official-reference or whole-strategy performance is produced.
"""

from dataclasses import dataclass
import math


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(name: str, value: float) -> float:
    result = _finite(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def futures_pnl_cny(quantity: float, multiplier: float, previous_price: float,
                    current_price: float, current_fx: float) -> float:
    """q * multiplier * (F_t - F_prev) * FX_t, for one unchanged contract.

    Never supply an old delivery contract as ``previous_price`` and a new
    delivery contract as ``current_price``. A roll requires two separate legs.
    Finite negative futures prices are algebraically supported; quote validity
    and the relevant product specification remain the acquisition layer's job.
    """
    q = _finite("quantity", quantity)
    point_value = _positive("multiplier", multiplier)
    before = _finite("previous_price", previous_price)
    after = _finite("current_price", current_price)
    fx = _positive("current_fx", current_fx)
    return q * point_value * (after - before) * fx


def futures_notional_cny(quantity: float, multiplier: float, price: float,
                         fx: float) -> float:
    """Signed marked notional q * multiplier * F * FX; this is not cash NAV."""
    return (_finite("quantity", quantity) * _positive("multiplier", multiplier)
            * _finite("price", price) * _positive("fx", fx))


def futures_trade_cost_cny(quantity_delta: float, multiplier: float,
                           trade_price: float, fx: float,
                           one_way_rate: float) -> float:
    """Cost on one executed buy or sell, using absolute traded notional."""
    rate = _finite("one_way_rate", one_way_rate)
    if rate < 0:
        raise ValueError("one_way_rate must be nonnegative")
    return abs(futures_notional_cny(quantity_delta, multiplier, trade_price, fx)) * rate


@dataclass(frozen=True)
class RollCost:
    exit_notional_cny: float
    entry_notional_cny: float
    exit_cost_cny: float
    entry_cost_cny: float

    @property
    def total_cost_cny(self) -> float:
        return self.exit_cost_cny + self.entry_cost_cny

    @property
    def gross_traded_notional_cny(self) -> float:
        return self.exit_notional_cny + self.entry_notional_cny


def futures_roll_cost_cny(old_quantity: float, old_multiplier: float,
                          old_exit_price: float, old_fx: float,
                          new_quantity: float, new_multiplier: float,
                          new_entry_price: float, new_fx: float,
                          one_way_rate: float) -> RollCost:
    """Close the old contract and open the new one; charge both gross legs.

    A different old/new delivery price is not roll P&L. Any old-contract P&L
    before its exit and new-contract P&L after its entry must be marked with
    ``futures_pnl_cny`` using their respective contracts and actual quantities.
    ``new_quantity`` may differ, e.g. to maintain a declared target notional.
    """
    exit_notional = abs(futures_notional_cny(old_quantity, old_multiplier,
                                            old_exit_price, old_fx))
    entry_notional = abs(futures_notional_cny(new_quantity, new_multiplier,
                                             new_entry_price, new_fx))
    exit_cost = futures_trade_cost_cny(-old_quantity, old_multiplier,
                                      old_exit_price, old_fx, one_way_rate)
    entry_cost = futures_trade_cost_cny(new_quantity, new_multiplier,
                                       new_entry_price, new_fx, one_way_rate)
    return RollCost(exit_notional, entry_notional, exit_cost, entry_cost)


def funded_asset_value_cny(quantity: float, price: float, fx: float) -> float:
    """Value of a funded native-currency asset; includes its principal FX mark."""
    return (_finite("quantity", quantity) * _positive("price", price)
            * _positive("fx", fx))


def funded_asset_pnl_cny(quantity: float, previous_price: float,
                         current_price: float, previous_fx: float,
                         current_fx: float) -> float:
    """Value change for an unchanged funded holding, excluding cash flows/fees.

    Use a separately specified treatment for dividends and corporate actions.
    A price index is a modeled funded index exposure, not itself a stock lot.
    """
    return (funded_asset_value_cny(quantity, current_price, current_fx)
            - funded_asset_value_cny(quantity, previous_price, previous_fx))


def funded_asset_return_cny(previous_price: float, current_price: float,
                            previous_fx: float, current_fx: float) -> float:
    """(P_t / P_prev) * (FX_t / FX_prev) - 1 for a funded position."""
    before = _positive("previous_price", previous_price)
    after = _positive("current_price", current_price)
    fx_before = _positive("previous_fx", previous_fx)
    fx_after = _positive("current_fx", current_fx)
    return (after / before) * (fx_after / fx_before) - 1
