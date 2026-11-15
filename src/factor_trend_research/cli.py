"""Choose a research module without importing optional modeling libraries."""
import argparse
import runpy
import sys
from . import __version__


def main():
    parser = argparse.ArgumentParser(description="Equity-factor and multi-asset trend research")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("task", choices=["task1", "task2"], help="Research module to run")
    # Leave a module's flags, including --help, to that module's own parser.
    if len(sys.argv) > 1 and sys.argv[1] in {"task1", "task2"}:
        task, remaining = sys.argv[1], sys.argv[2:]
    else:
        args = parser.parse_args()
        task, remaining = args.task, []
    sys.argv = [f"factor-trend-research {task}", *remaining]
    runpy.run_module(f"factor_trend_research.{task}", run_name="__main__")
