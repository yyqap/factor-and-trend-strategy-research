"""Long-only, fractional-unit accounting core for native futures and funded indices.

This module does not load data, generate signals, run the 16-asset strategy, or
report strategy performance. Five modeled funded index exposures may be mixed
with any number of native futures legs; their accounting remains separate.

Input and execution conventions
-------------------------------
* Dates are already resolved exchange-session dates, supplied as YYYY-MM-DD.
  The caller resolves time zones, holidays, quote freshness and FX availability.
* A daily mark supplies the SAME held futures contract at both interval ends.
  At the interval end, a separately quoted next contract can be opened. A roll
  gap is never investment P&L. Raw prices must be numeric quote points, with an
  explicitly supplied native-currency point value (e.g. ZT=2000, ZF/ZN=1000).
* Interval P&L belongs to the positions carried into the interval. Funded
  positions include the FX change on their actual principal. Futures have no
  invested principal: q * point_value * (end_price-start_price) is translated
  at interval-end FX and immediately converted into CNY cash. No USD cash is
  retained, so USD collateral/funding and its FX exposure are not simulated.
* Rebalancing happens at the supplied end prices AFTER interval P&L. The
  caller supplies targets decided without using unavailable information. New
  positions first earn P&L in the following interval. Exact close fills and
  simultaneous CNY conversion are research conventions, not an execution model.
* Target weights mean funded value/NAV OR futures marked notional/NAV. Cash is
  NAV minus funded market value, not NAV minus all risky notional exposures.
  Futures margin is part of cash, not an additional asset or an expense.
* Costs are explicit assumed one-way fractions of gross traded notional. A
  native roll has a close and an open leg, each charged once. Post-cost NAV is
  solved before turning target weights into units. No commission, tax, spread,
  impact, margin-call, integer-lot or collateral-interest model is implied.
* A non-tradeable instrument keeps its units and its held contract. Requested
  changes are reported as deferred. Missing roll overlap must be resolved by
  acquisition; it is never silently filled with the next contract's price.
* Without targets, funded units and futures quantities are fixed, except an
  allowed mandatory roll or explicitly supplied absolute basket notionals.
  A roll preserves marked notional by
  default; choose roll_policy='quantity' to preserve the contract count.
* Index exposures are modeled funded positions, not executable index lots.
  Their input must explicitly declare price/total-return and dividend treatment.

All prices used for notional sizing must be positive. This deliberately rejects
negative-price contracts rather than silently choosing an ambiguous weight rule.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal, Mapping
import math


def _number(name, value, *, positive=False, nonnegative=False):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _session_date(value):
    if isinstance(value, datetime):
        raise ValueError("Supply an already resolved session date, not a timestamp")
    if isinstance(value, date):
        return value
    if isinstance(value, str) and len(value) == 10:
        return date.fromisoformat(value)
    raise ValueError("Session dates must be date objects or YYYY-MM-DD strings")


@dataclass(frozen=True)
class Instrument:
    asset_id: str
    kind: Literal["funded", "future"]
    currency: str
    point_value: float = 1.0
    one_way_cost_rate: float = 0.0005

    def __post_init__(self):
        if not self.asset_id or self.kind not in ("funded", "future") or not self.currency:
            raise ValueError("Instrument needs an id, kind and currency")
        point_value = _number("point_value", self.point_value, positive=True)
        rate = _number("one_way_cost_rate", self.one_way_cost_rate, nonnegative=True)
        if rate >= 1:
            raise ValueError("one_way_cost_rate must be below one")
        if self.kind == "funded" and self.point_value != 1:
            raise ValueError("Funded units use point_value=1; scale units explicitly")
        object.__setattr__(self, "point_value", point_value)
        object.__setattr__(self, "one_way_cost_rate", rate)


@dataclass(frozen=True)
class IntervalMark:
    start_price: float
    end_price: float
    start_fx: float = 1.0
    end_fx: float = 1.0
    held_contract: str | None = None
    next_contract: str | None = None
    next_price: float | None = None
    tradeable: bool = True

    def __post_init__(self):
        for name in ("start_price", "end_price", "start_fx", "end_fx"):
            object.__setattr__(self, name, _number(name, getattr(self, name), positive=True))
        if self.next_price is not None:
            object.__setattr__(self, "next_price", _number("next_price", self.next_price, positive=True))


@dataclass(frozen=True)
class LastMark:
    price: float
    fx: float
    contract: str | None = None


@dataclass(frozen=True)
class PortfolioState:
    cash_cny: float
    funded_units: Mapping[str, float] = field(default_factory=dict)
    future_quantities: Mapping[str, float] = field(default_factory=dict)
    future_contracts: Mapping[str, str] = field(default_factory=dict)
    last_marks: Mapping[str, LastMark] = field(default_factory=dict)
    as_of: date | str | None = None


@dataclass(frozen=True)
class Trade:
    asset_id: str
    kind: str
    action: str
    contract: str | None
    quantity_delta: float
    price: float
    point_value: float
    fx: float
    gross_notional_cny: float
    cost_cny: float


@dataclass(frozen=True)
class DayResult:
    state: PortfolioState
    opening_nav_cny: float
    funded_pnl_cny: Mapping[str, float]
    futures_vm_native: Mapping[str, float]
    futures_vm_cny: Mapping[str, float]
    management_fee_cny: float
    pretrade_nav_cny: float
    transaction_cost_cny: float
    end_nav_cny: float
    funded_market_value_cny: Mapping[str, float]
    futures_notional_cny: Mapping[str, float]
    funded_weights: Mapping[str, float]
    futures_notional_weights: Mapping[str, float]
    cash_weight: float
    trades: tuple[Trade, ...]
    deferred: tuple[str, ...]

    @property
    def gross_traded_notional_cny(self):
        return sum(t.gross_notional_cny for t in self.trades)


def expand_target_weights(logical_weights, logical_to_legs, instruments):
    """Expand, e.g., one black-metals weight into five declared actual legs.

    Example mapping: {'black': {'RB': .2, 'HC': .2, 'I': .2, 'J': .2,
    'JM': .2}, 'CSI300': {'CSI300': 1.0}}. Each supplied logical weight is
    preserved; residual cash is not allocated or normalized away. Missing
    logical targets mean zero, and every actual instrument is returned.
    """
    ids = {x.asset_id for x in instruments}
    if len(ids) != len(instruments):
        raise ValueError("Duplicate actual instrument ids")
    unknown = set(logical_weights) - set(logical_to_legs)
    if unknown:
        raise ValueError(f"Unknown logical target ids: {sorted(unknown)}")
    result = dict.fromkeys(ids, 0.0)
    for logical, legs in logical_to_legs.items():
        if not legs or set(legs) - ids:
            raise ValueError(f"Invalid actual-leg mapping for {logical}")
        fractions = [_number("leg fraction", v, nonnegative=True) for v in legs.values()]
        if not math.isclose(sum(fractions), 1.0, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Leg fractions must add to one: {logical}")
        weight = _number(f"weight {logical}", logical_weights.get(logical, 0), nonnegative=True)
        for actual, fraction in legs.items():
            result[actual] += weight * fraction
    return result


def account_interval(state, instruments, marks, *, start_date, end_date,
                     target_weights=None, management_fee_cny=0.0,
                     roll_policy="notional", target_gross_limit=1.0,
                     fixed_notional_targets_cny=None):
    """Mark old positions, then execute allowed EOD trades; return a NEW state.

    ``marks`` has exactly one IntervalMark per actual Instrument. For a future,
    held_contract is the contract held during the interval. next_contract and
    next_price describe the independently quoted EOD roll destination. When no
    roll is requested, next_contract=None (or the same id) and next_price=None.
    ``target_weights`` is a full actual-leg target vector; missing ids are zero.
    If it is None there is no allocation rebalance, only eligible native rolls.
    Optional ``fixed_notional_targets_cny`` supplies absolute CNY notional for
    specified future legs, overriding their weight target. This can preserve a
    basket's pre-fee total notional while equalizing only its component legs;
    other holdings keep their units when target_weights is None. Cash pays the
    resulting costs. These absolute targets do not depend on the NAV solver.

    A closed instrument is frozen even if supplied a new target. If this makes
    the trades in other funded instruments unaffordable, reject the plan rather
    than borrow or silently scale it. The input state is never mutated.
    """
    start, end = _session_date(start_date), _session_date(end_date)
    if end <= start:
        raise ValueError("End session must follow start session")
    if state.as_of is not None and _session_date(state.as_of) != start:
        raise ValueError("Interval start does not match state's as_of date")
    if roll_policy not in ("notional", "quantity"):
        raise ValueError("roll_policy must be notional or quantity")
    specs = {x.asset_id: x for x in instruments}
    if not specs or len(specs) != len(instruments):
        raise ValueError("Need unique instrument ids")
    if set(marks) != set(specs):
        raise ValueError("Marks must cover exactly the actual instrument ids")
    for values in (state.funded_units, state.future_quantities,
                   state.future_contracts, state.last_marks):
        if set(values) - set(specs):
            raise ValueError("State contains unknown instruments")
    cash = _number("cash_cny", state.cash_cny, nonnegative=True)
    fee = _number("management_fee_cny", management_fee_cny, nonnegative=True)
    limit = _number("target_gross_limit", target_gross_limit, positive=True)
    targets = None
    fixed = {}
    if fixed_notional_targets_cny is not None:
        if set(fixed_notional_targets_cny) - set(specs):
            raise ValueError("Fixed-notional targets contain unknown instruments")
        for a, value in fixed_notional_targets_cny.items():
            if specs[a].kind != "future":
                raise ValueError("Absolute notional targets are reserved for future legs")
            fixed[a] = _number(f"fixed notional {a}", value, nonnegative=True)
    if target_weights is not None:
        if set(target_weights) - set(specs):
            raise ValueError("Targets contain unknown actual instruments")
        targets = {a: _number(f"target {a}", target_weights.get(a, 0), nonnegative=True)
                   for a in specs}
        if sum(targets.values()) > limit + 1e-12:
            raise ValueError("Target gross notional exceeds the declared limit")
        # Guarantees a strictly increasing post-cost NAV root function.
        if sum(targets[a] * specs[a].one_way_cost_rate for a in specs) >= 1:
            raise ValueError("Target and cost rates cannot support a unique NAV solve")

    old_funded, old_futures = {}, {}
    funded_pnl, vm_native, vm_cny = {}, {}, {}
    funded_end, destination, destination_price = {}, {}, {}
    opening_funded = 0.0
    deferred = set()
    for a, spec in specs.items():
        m = marks[a]
        if not isinstance(m, IntervalMark) or not isinstance(m.tradeable, bool):
            raise ValueError("Each mark needs an IntervalMark and a boolean tradeable mask")
        for name in ("start_price", "end_price", "start_fx", "end_fx"):
            _number(f"{a}.{name}", getattr(m, name), positive=True)
        if spec.currency == "CNY" and (m.start_fx != 1 or m.end_fx != 1):
            raise ValueError("Native CNY instrument must use FX=1")
        if spec.kind == "funded":
            if (a in state.future_quantities or a in state.future_contracts
                    or m.held_contract is not None or m.next_contract is not None
                    or m.next_price is not None):
                raise ValueError("Funded asset must not have futures-contract fields")
            q = _number(f"{a} funded units", state.funded_units.get(a, 0), nonnegative=True)
            old_funded[a] = q
            before = q * m.start_price * m.start_fx
            after = q * m.end_price * m.end_fx
            opening_funded += before
            funded_end[a] = after
            funded_pnl[a] = after - before
            destination[a], destination_price[a] = None, m.end_price
        else:
            if a in state.funded_units or not isinstance(m.held_contract, str) or not m.held_contract:
                raise ValueError("Future needs a nonempty held_contract and no funded units")
            q = _number(f"{a} futures quantity", state.future_quantities.get(a, 0), nonnegative=True)
            held = state.future_contracts.get(a)
            if q != 0 and held != m.held_contract:
                raise ValueError(f"{a}: interval does not mark the held contract")
            old_futures[a] = q
            vm_native[a] = q * spec.point_value * (m.end_price - m.start_price)
            vm_cny[a] = vm_native[a] * m.end_fx
            next_contract = m.next_contract or m.held_contract
            if not isinstance(next_contract, str) or not next_contract:
                raise ValueError("Invalid next contract id")
            if next_contract == m.held_contract:
                if m.next_price is not None and not math.isclose(
                        m.next_price, m.end_price, rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError("Same-contract EOD quote cannot have two different prices")
                next_price = m.end_price
            elif m.tradeable:
                if m.next_price is None:
                    raise ValueError(f"{a}: a roll needs both old and new EOD prices")
                next_price = _number(f"{a}.next_price", m.next_price, positive=True)
            else:
                next_price = m.end_price
                if q != 0:
                    deferred.add(a)
            if not m.tradeable:
                next_contract = m.held_contract
            destination[a], destination_price[a] = next_contract, next_price
        previous = state.last_marks.get(a)
        # A flat futures instrument may select another contract without trading.
        # Nonzero carried holdings MUST have a continuous same-contract mark.
        compare_previous = (previous is not None and
                            (spec.kind == "funded" or previous.contract == m.held_contract
                             or old_futures.get(a, 0) != 0))
        if compare_previous:
            if spec.kind == "future" and previous.contract != m.held_contract:
                raise ValueError(f"{a}: missing post-roll held-contract continuity")
            if not math.isclose(previous.price, m.start_price, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"{a}: interval start price differs from prior end mark")
            if not math.isclose(previous.fx, m.start_fx, rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError(f"{a}: interval start FX differs from prior end FX")

    opening_nav = cash + opening_funded
    _number("opening NAV", opening_nav, positive=True)
    pretrade_cash = cash + sum(vm_cny.values()) - fee
    pretrade_nav = pretrade_cash + sum(funded_end.values())
    _number("pretrade NAV", pretrade_nav, positive=True)

    def plan(post_cost_nav):
        funded_q, future_q, trades = {}, {}, []

        def add_trade(a, action, contract, delta, price):
            if delta == 0:
                return
            spec, m = specs[a], marks[a]
            gross = abs(delta) * spec.point_value * price * m.end_fx
            trades.append(Trade(a, spec.kind, action, contract, delta, price,
                                spec.point_value, m.end_fx, gross,
                                gross * spec.one_way_cost_rate))

        for a, spec in specs.items():
            m = marks[a]
            prior = old_funded[a] if spec.kind == "funded" else old_futures[a]
            price = destination_price[a]
            if m.tradeable and a in fixed:
                quantity = fixed[a] / (price * spec.point_value * m.end_fx)
            elif m.tradeable and targets is not None:
                quantity = targets[a] * post_cost_nav / (price * spec.point_value * m.end_fx)
            elif (m.tradeable and spec.kind == "future"
                  and destination[a] != m.held_contract and roll_policy == "notional"):
                quantity = prior * m.end_price / price
            else:
                quantity = prior
            if spec.kind == "funded":
                funded_q[a] = quantity
                add_trade(a, "rebalance", None, quantity - prior, m.end_price)
            else:
                future_q[a] = quantity
                if destination[a] != m.held_contract:
                    add_trade(a, "roll_close", m.held_contract, -prior, m.end_price)
                    add_trade(a, "roll_open", destination[a], quantity, price)
                else:
                    add_trade(a, "rebalance", m.held_contract, quantity - prior, price)
        return funded_q, future_q, trades

    # Fees depend on target units, which depend on post-fee NAV. Bisection also
    # handles selling funded assets; it does not assume turnover is buy-only.
    def residual(value):
        return value + sum(t.cost_cny for t in plan(value)[2]) - pretrade_nav

    lo, hi = 0.0, pretrade_nav
    tolerance = 1e-11 * max(1.0, pretrade_nav)
    if residual(lo) > tolerance:
        raise ValueError("Mandatory transaction costs exceed available NAV")
    for _ in range(100):
        mid = (lo + hi) / 2
        if residual(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-13 * max(1.0, pretrade_nav):
            break
    solved_nav = (lo + hi) / 2
    funded_q, future_q, trades = plan(solved_nav)
    cost = sum(t.cost_cny for t in trades)
    funded_cash_flow = sum((funded_q[a] - old_funded[a])
                           * marks[a].end_price * marks[a].end_fx for a in funded_q)
    end_cash = pretrade_cash - funded_cash_flow - cost
    if end_cash < -tolerance:
        raise ValueError("Frozen funded holdings make the requested plan unaffordable")
    if end_cash < 0:
        end_cash = 0.0  # Only floating-point dust within the stated tolerance.
    funded_values = {a: q * marks[a].end_price * marks[a].end_fx for a, q in funded_q.items()}
    future_notionals = {a: q * specs[a].point_value * destination_price[a] * marks[a].end_fx
                        for a, q in future_q.items()}
    end_nav = end_cash + sum(funded_values.values())
    if abs(end_nav - (pretrade_nav - cost)) > tolerance or abs(end_nav - solved_nav) > tolerance:
        raise AssertionError("Cash + funded value must reconcile to post-cost NAV")
    if abs(end_nav - opening_nav - sum(funded_pnl.values())
           - sum(vm_cny.values()) + fee + cost) > tolerance:
        raise AssertionError("Interval P&L and actual costs do not reconcile")
    _number("end NAV", end_nav, positive=True)
    if targets is not None:
        for a, spec in specs.items():
            if not marks[a].tradeable:
                actual = funded_values.get(a, future_notionals.get(a, 0)) / end_nav
                if abs(actual - targets[a]) > 1e-10:
                    deferred.add(a)
    for a in fixed:
        if not marks[a].tradeable and abs(future_notionals[a] - fixed[a]) > tolerance:
            deferred.add(a)
    last_marks = {a: LastMark(destination_price[a], marks[a].end_fx, destination[a]) for a in specs}
    new_state = PortfolioState(end_cash, dict(funded_q), dict(future_q),
                               {a: destination[a] for a in future_q}, last_marks, end)
    return DayResult(new_state, opening_nav, funded_pnl, vm_native, vm_cny, fee,
                     pretrade_nav, cost, end_nav, funded_values, future_notionals,
                     {a: value / end_nav for a, value in funded_values.items()},
                     {a: value / end_nav for a, value in future_notionals.items()},
                     end_cash / end_nav, tuple(trades), tuple(sorted(deferred)))
