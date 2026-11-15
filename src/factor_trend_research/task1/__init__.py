"""Reproducible cross-sectional factor research with explicit data boundaries."""

from .data import load_panel, synthetic_panel, validate_panel
from .features import screen_features, tail_labels
from .portfolio import backtest, performance_metrics
from .research import run_research

__all__ = ["load_panel", "synthetic_panel", "validate_panel", "screen_features",
           "tail_labels", "backtest", "performance_metrics", "run_research"]
