"""Declared16 logical exposures and20 original-underlying accounting legs.

Contract point values are configuration for this research design, not a feed
of current exchange specifications. Real inputs must confirm their quote units.
"""
from .native_engine import Instrument

FUNDED = ("CSI300", "CSI500", "CSI1000", "Nasdaq100", "SP500")
FUTURES = ("TS", "TF", "T", "ZT", "ZF", "ZN", "B", "AU", "CU", "M", "RB", "HC", "I", "J", "JM")
POINTS = {"TS": 20000, "TF": 10000, "T": 10000, "ZT": 2000, "ZF": 1000, "ZN": 1000,
          "B": 1000, "AU": 1000, "CU": 5, "M": 10, "RB": 10, "HC": 10, "I": 100, "J": 100, "JM": 60}
USD = frozenset(("Nasdaq100", "SP500", "ZT", "ZF", "ZN", "B"))
LOGICAL = {
    **{a: {a: 1.0} for a in FUNDED},
    "China2Y": {"TS": 1.0}, "China5Y": {"TF": 1.0}, "China10Y": {"T": 1.0},
    "US2Y": {"ZT": 1.0}, "US5Y": {"ZF": 1.0}, "US10Y": {"ZN": 1.0},
    "Brent": {"B": 1.0}, "ShanghaiGold": {"AU": 1.0}, "ShanghaiCopper": {"CU": 1.0},
    "SoybeanMeal": {"M": 1.0}, "FerrousComplex": {a: 0.2 for a in ("RB", "HC", "I", "J", "JM")},
}
CAPS = dict(zip(LOGICAL, (0.1, 0.1, 0.1, 0.1, 0.1, 0.4, 0.4, 0.4, 0.6, 0.4, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1)))


def instruments(cost_rate=0.0005):
    return ([Instrument(a, "funded", "USD" if a in USD else "CNY", 1, cost_rate) for a in FUNDED]
            + [Instrument(a, "future", "USD" if a in USD else "CNY", POINTS[a], cost_rate) for a in FUTURES])
