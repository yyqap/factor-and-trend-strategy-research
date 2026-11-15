"""Task2: native multi-asset trend research and explicit synthetic fixtures."""
from .native_engine import Instrument, IntervalMark, PortfolioState, account_interval
from .native_strategy import DEFAULT_CONFIG, SignalFrames, run_native_strategy
from .native_signals import build_signals

__all__ = ["Instrument", "IntervalMark", "PortfolioState", "account_interval",
           "DEFAULT_CONFIG", "SignalFrames", "run_native_strategy", "build_signals"]
