# Task2: native multi-asset trend research

This module is a portable version of the later original-underlying research
engine. It is a GMAT-style research prototype, not a replication of an official
index or a claim to know its undisclosed implementation. There is one declared
configuration, not an optimizer selected using the displayed return.

## What the public demo does

```bash
python -m factor_trend_research.task2 --demo --output outputs/task2_demo
```

The demo generates **synthetic prices, FX, exchange-session masks and delivery
contracts**. It runs the16-logical/20-leg software path, including contract
rolls, funded positions, variation margin and transaction costs. Its CSVs and
summary are software fixtures, **not historical investment results**. Generated
weekdays are not actual exchange calendars. Seed changes are not strategy
research or performance optimization.

`--config configs/task2.json` loads the declared rules/caps. The default demo has
520 generated observations; the first300 supply a warm-up buffer. `--seed` and
`--demo-periods` affect the generated fixture only. All files are written to the
requested output directory; no market data is downloaded.

## Declared universe

There are16 logical exposures:

- Five modeled funded price indices: CSI300, CSI500, CSI1000, Nasdaq100 and SP500.
- Chinese government bond futures: China2Y/TS, China5Y/TF and China10Y/T.
- US Treasury futures: US2Y/ZT, US5Y/ZF and US10Y/ZN.
- Brent/B, ShanghaiGold/AU, ShanghaiCopper/CU and SoybeanMeal/M.
- FerrousComplex: RB, HC, I, J and JM, each20% of the group's marked notional.

The ferrous basket expands one logical exposure into five futures legs, giving
**five funded indices plus fifteen futures =20 actual accounting legs**. The
funded indices exclude dividends in this version; they are modeled index
exposures, not ETF fills or executable index lots. ETFs cannot silently replace
the original futures or indices in the native loader.

Point values in `universe.py` are native-currency P&L per one numeric quote point
per contract: TS20000; TF/T10000; ZT2000; ZF/ZN1000; Brent1000; AU1000; CU5;
M/RB/HC10; I/J100; JM60. These describe the research's quote-unit contract, not a
current exchange-specification feed. Real input providers must confirm units,
including parsing US Treasury fractional quotes into numeric points first.

## Signals and availability

Trend signals use **local-currency observations**. Futures signal levels are
forward chains of same-held-contract returns, so an old/new delivery-price gap
does not become momentum. The accounting engine never treats that chain as an
invested asset. Portfolio-risk returns are kept separately on a CNY basis.

On each asset's own observed sessions, the four components are:

1. 21-session return divided by its21-session annualized volatility.
2. 252-session return divided by its252-session annualized volatility.
3. The unadjusted252-session return.
4. Position within the trailing252-session low/high range.

Each component becomes a cross-sectional percentile rank; their equal-weight
mean is the asset score. A flat high/low range receives position0.5. An asset
needs260 valid native returns (261 observed endpoints). Holidays are not
manufactured observations. A genuinely missing return resets the signal chain
and its warm-up; later missing data cannot erase earlier valid signal frames.

Already-known native metrics are mapped forward to the joint valuation calendar
without filling through active missing-data barriers. For the five-leg basket,
an observed signal endpoint requires all five constituent observations.
Historical source data must separately establish which absences are confirmed
closures. An arbitrary weekday calendar is not a substitute for that evidence.

## Top11, volatility sizing and caps

The allocator selects the highest11 scores, using stable ordering for ties.
Its initial selected-asset target is `(0.08 / 11) / sigma`, subject to per-asset
caps and a total risky-notional target of at most0.995 of NAV. Residual capacity
remains unallocated; capped weights are never normalized back to one.

Ordinarily `sigma` is the maximum of22-,65- and130-session local annualized
volatility. If trailing252-session return/short-volatility correlation is
negative, the declared adaptive rule uses21-session volatility instead.
The correlation needs at least130 valid observations; the volatility floor is
0.005. This is inverse-volatility sizing with an equal coefficient, **not
covariance-based equal-risk-contribution optimization**.

Caps are10% for the five equities and commodities/basket,40% for China2Y/5Y/10Y
and US5Y/10Y, and60% for US2Y. `configs/task2.json` records the complete mapping.
Caps constrain targets/executed rebalances. Holdings drift between trades, and
a closed market can delay a requested reduction; continuous cap enforcement is
not claimed.

## Reversal rule

The native primary confirmation is **price above its five-session moving
average**, not the earlier five-day-return alternative. The rule first observes
an adjacent-MA alignment score above0.35. Alignment compares MA1≥MA2 through
MA259≥MA260. During a subsequent weak episode, its minimum must reach at most
0.10, and the current value must recover at least0.005 from that minimum while
price is above MA5.

Triggered assets must already be in the selected group. Their allocation may
increase by up to five percentage points, subject to caps. Other selected
holdings fund the increase proportionally; gross exposure is preserved. This
boost is a disclosed prototype assumption, not a confirmed official-index rule.
Reversal changes allocation after trend ranking; it is not an extra score term.

## Portfolio risk and execution timing

For a candidate target, estimated portfolio volatility is the largest sample
standard deviation of the22/65/130 trailing CNY exposure-return paths multiplied
by `sqrt(252)`. These are hypothetical current weights applied to past returns,
not an estimate that guarantees future strategy volatility.

- Above4.5%, the daily rule scales toward4%.
- At the monthly target update, below3% can scale toward4%, respecting caps.
- Monthly execution is staggered over common sessions4–7. Each tranche closes
  `1 / remaining_tranches` of the gap from current exposure to the fixed target.
- If session4 is unready or the complete four-session phase does not fit the
  declared evaluation window, that month is skipped rather than starting late.
- Decisions use at least one complete joint-valuation-date lag. Close rebalances
  happen **after** the carried holdings earn the current interval's P&L; new
  holdings first earn P&L in the next interval.

Allocation trades require every actual leg's declared trading mask to be open.
Eligible contract rolls and internal ferrous-basket maintenance can occur on
their own native sessions. The common-session schedule is an input calendar,
not something inferred from favorable prices. Simultaneous CN/US/ICE close
sizing is a synchronous research approximation, not a claim about executable
cross-time-zone fills.

## Native accounting and costs

For an unchanged funded position, CNY P&L is
`q * (P_end * FX_end - P_start * FX_start)`.
For a futures position, variation margin is
`q * point_value * (F_end - F_start) * FX_end`, with the **same held contract**
at both endpoints. Futures have no imaginary invested principal: a flat futures
price with moving FX produces no variation-margin P&L. USD variation margin is
immediately converted to CNY; no USD cash/collateral account is retained.

NAV is **CNY cash plus funded market value**, not cash plus futures notional.
Opening a futures exposure does not spend its marked notional principal.
Futures margin is not modeled as an expense or a second asset.

A roll marks the old contract's P&L, then closes it and opens an independently
quoted destination contract. The delivery-price difference itself is not P&L.
The default quantity adjustment preserves marked notional across the roll.
The ferrous basket is rebalanced to equal component notionals on eligible
sessions, preserving the pre-fee group notional unless outer allocation changes.

Assumed one-way transaction cost is5bps of **actual gross traded notional**;
each roll's close and open legs are charged. Targets are sized on post-cost NAV
using a monotone solve. Management fee is0.5% per year, accrued continuously
over actual elapsed calendar days/365.25. CNY cash earns zero in this version.
No broker commission schedule, tax, spread/impact, integer lots, margin calls,
collateral interest, borrowing or executable capacity is inferred.

## Local real-data input contract

```bash
python -m factor_trend_research.task2 \
  --native-inputs /path/to/private/prepared_inputs \
  --output outputs/task2_native
```

No financial raw inputs, archives or vendor quotes are bundled. A native run
requires these five local CSVs, plus `metadata.json`:

- `accounting_intervals.csv`: date, asset_id, start_price, end_price, start_fx,
  end_fx, held_contract, next_contract, next_price, tradeable. Each interval has
  all20 actual legs. Old and next contract marks are distinct fields.
- `local_actual_leg_returns.csv`: date-indexed20-leg same-contract local returns.
- `cny_actual_leg_risk_returns.csv`: aligned20-leg CNY exposure returns. Funded
  returns include principal FX; futures risk returns convert only the price move.
- `native_observed.csv`: aligned20-leg explicit true/false observation masks.
- `common_execution_dates.csv`: date and explicit true/false `common_open`.

Dates are unique, increasing, date-only resolved sessions. Mark prices and FX
must be finite/positive. Only the first normalization return may be missing in
a complete prepared window; active internal risk gaps block the strategy.
The loader checks return/mark consistency, full leg coverage and common-calendar
agreement. It does not fill missing prices or assume absent quotes are holidays.

Metadata must declare:

- `schema_version:1`, `input_kind:"native_original_underlyings"` and explicit
  `data_gate_passed:true`.
- `evaluation_start`, `evaluation_end`, and written `calendar_policy`,
  `fx_policy`, `contract_roll_policy`.
- All20 `instruments`, each with asset_id, kind, currency, point_value, source,
  price_type and representation. Funded representation is `price_index` with
  `dividend_policy:"excluded"`; futures representation is
  `native_futures_contract`.
- `input_sha256`, mapping each of the five exact CSV filenames to its SHA256.

A declared passed gate is **not independent vendor or point-in-time
certification**. The caller must validate history, quote units, FX availability,
holiday evidence and roll overlap before writing the gate. A historical FX
observation date alone does not prove its release timestamp was known then.
The public code does not recreate the private acquisition archive automatically.

## Outputs and checks

The runner saves daily NAV/cash, logical exposures, actual quantities/contracts,
trades/costs, native observation counts, lagged decisions and a labeled summary.
It reconciles the NAV change to funded P&L plus futures VM minus both fee types,
and checks NAV equals cash plus funded assets.

Metrics use compounded NAV, actual elapsed years for CAGR, daily sample standard
deviation times`sqrt(252)` for volatility, arithmetic daily mean/std times
`sqrt(252)` for zero-RF Sharpe, and peak-to-trough NAV drawdown including initial
NAV. A daily observation MDD does not measure intraday drawdown. Synthetic
metrics remain synthetic regardless of their apparent quality.

Tests use generated/golden-ledger examples, not vendor data. They check native
observation warm-up, future-data mutation invariance, next-interval P&L timing,
reversal funding, caps/residual cash, post-cost sizing, closed-market deferral,
same-contract roll continuity, two-sided roll fees, and the difference between
funded FX exposure and futures variation margin.

## Mapping the public code to the saved historical results

`results/task2/` contains a **completed native-underlying historical run**, not
the generated demo. Its initial NAV date is2020-06-03, first execution is
2020-06-04, and cutoff is2023-08-18. The source search began in2018, but missing
early source/contract observations were not backfilled. Evaluation started only
after the complete input suffix, native-observation warm-up and the first full
monthly execution phase were available.

The correspondence is:

- `native_signals.py`: native-session return chains, four trend components,
  percentile scores, volatility/correlation and MA5-confirmed reversal flags.
- `rules.py`: Top11 selection, inverse-volatility target, caps, reversal-funded
  additions and the22/65/130-session portfolio risk rules.
- `native_strategy.py`: lagged decisions, monthly sessions4–7, exposure drift,
  common-session allocation and native ferrous-basket maintenance.
- `native_engine.py`: funded units, futures quantities, cash/variation margin,
  roll continuity, traded-notional costs and post-cost position sizing.
- `runner.py::metrics`: actual-calendar-time CAGR, zero-RF arithmetic Sharpe and
  daily-observation NAV MDD; these match the saved headline metrics.
- `future_accounting.py`: separately tested P&L/FX/roll-cost primitives. The
  portfolio engine implements the same identities in its complete state ledger.

Before publication, the portable core was rerun locally on the original
hash-approved prepared input bundle, reading private inputs in place. **96
correspondence checks passed**: all836 valuation dates, every daily accounting
column, all logical weights, all actual positions/trades and every decision
matched the canonical native run. Public normalized NAV and net-return rows
matched, as did both saved performance windows. The largest absolute difference
among daily accounting columns was approximately`2.33e-10`CNY.

The full-window saved/recomputed CAGR is2.5388%, zero-RF Sharpe0.937998 and NAV
MDD3.4408%. These are conditional research results under the assumptions above,
not executable investment returns. The836 union-calendar marks span about3.206
calendar years; the fixed`sqrt(252)` convention is retained rather than claiming
it equals the observed annual mark frequency. A separate2021-to-cutoff window
is identified in the saved performance table and is not mixed with the full run.

The engine and futures-primitives source matched the canonical implementations;
signal, allocation and execution functions matched structurally after portable
imports and an explicit caller-gate labeling change. The public test suite for
Task2 passed43 tests plus7 parameterized subtests in the publication environment.

### Exact-reproduction boundary

The canonical historical driver consumed **precomputed, as-of native signal
frames** plus a verified complete CNY-risk suffix. It retained the earlier
signal history and missing-episode resets. The general`--native-inputs`CLI
instead rebuilds signal frames from the caller's five-CSV input contract.
Merely cropping raw prices to the evaluation start or changing the warm-up
window can change signal/reversal state; that does not reproduce the saved run.

The lower-level`run_native_strategy`API accepts an explicitly constructed
`SignalFrames`object for the canonical prepared-frame path. Exact historical
reproduction requires the same approved20-leg marks, CNY-risk matrix,
observation/signal history, common calendar, contract/FX conventions, date gate,
configuration and initial state. Those financial source archives and private
approval manifests are not redistributed. The local correspondence run does
not re-audit vendor publication timestamps or establish official-index
replication. The synthetic demo is independently runnable without those inputs.
