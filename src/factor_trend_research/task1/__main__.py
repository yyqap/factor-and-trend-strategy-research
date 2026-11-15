"""Run a portable synthetic fixture or research on a user-owned local panel."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .data import load_panel, synthetic_panel
from .diagnostics import asof_industries, industry_rank_ic, industry_time_inference
from .portfolio import backtest
from .research import load_config, monthly_rank_ic, run_research


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="Synthetic software fixture, NOT historical market results")
    source.add_argument("--input", type=Path, help="Normalized panel CSV or original numbered CSV directory")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--models", help="Comma-separated model families; demo defaults ridge,logistic")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, help="Matched holding-period benchmark and RF CSV")
    parser.add_argument("--confirm-matched-horizon", action="store_true",
                        help="Confirm target=stock minus benchmark arithmetic return over provided dates")
    parser.add_argument("--industry-history", type=Path, help="Optional dated historical classifications")
    args = parser.parse_args(argv)
    if args.benchmark is not None and not args.confirm_matched_horizon:
        parser.error("--benchmark requires --confirm-matched-horizon after verifying source semantics and dates")
    config = load_config(args.config)
    families = args.models.split(",") if args.models else (["ridge", "logistic"] if args.demo else config["models"])
    panel, features, quality = synthetic_panel() if args.demo else load_panel(args.input)
    with threadpool_limits(limits=1):
        results = run_research(panel, features, config, families)
    args.output.mkdir(parents=True, exist_ok=True)
    for key in ["cv_folds", "leaderboard", "feature_screening"]:
        results[key].to_csv(args.output/f"{key}.csv", index=False)
    results["predictions"].to_csv(args.output/"predictions.csv.gz", index=False, compression="gzip")
    (args.output/"fit_logs.json").write_text(json.dumps(_json_safe(results["fit_logs"]), indent=2)+"\n")
    benchmark = pd.read_csv(args.benchmark) if args.benchmark is not None else None
    metrics = []
    for n in config["portfolio_sizes"]:
        for weighting in config["weighting_rules"]:
            if weighting == "risk_rank" and (config["risk_feature"] not in panel or panel[config["risk_feature"]].isna().any()):
                # This is an explicitly unavailable portfolio, never a substitute weighting rule.
                print("SKIP risk_rank: missing risk characteristic; equal-weight portfolios remain available")
                continue
            for cost in config["cost_bps"]:
                monthly, holdings, metric = backtest(results["predictions"], n=n, weighting=weighting,
                    cost_bps=cost, risk_feature=config["risk_feature"], benchmark=benchmark)
                name = f"top{n}_{weighting}_{cost}bps"
                monthly.to_csv(args.output/f"monthly_{name}.csv", index=False)
                if cost == 10:
                    holdings.to_csv(args.output/f"holdings_{name}.csv.gz", index=False, compression="gzip")
                metrics.append(metric)
    pd.concat(metrics, ignore_index=True).to_csv(args.output/"portfolio_metrics.csv", index=False)
    signal_metrics = []
    for model, group in results["predictions"].groupby("model", sort=True):
        ic = monthly_rank_ic(group, group.score)
        signal_metrics.append({"model": model, "months": len(ic), "mean_rank_ic": float(ic.rank_ic.mean()),
                               "positive_ic_month_fraction": float(ic.rank_ic.gt(0).mean())})
    if args.industry_history is not None:
        joined = asof_industries(results["predictions"], pd.read_csv(args.industry_history, dtype={"stock": str}))
        industry = pd.concat([industry_rank_ic(joined, minimum_group=n) for n in [5, 2, 10]], ignore_index=True)
        industry.to_csv(args.output/"industry_rank_ic.csv", index=False)
        inference, bootstrap = industry_time_inference(industry)
        inference.to_csv(args.output/"industry_inference.csv", index=False)
        bootstrap.to_csv(args.output/"industry_bootstrap.csv", index=False)
    summary = {"data": quality, "config": config, "models_run": families, "selection": results["selection"],
               "signal_metrics": signal_metrics, "absolute_metrics_available": benchmark is not None,
               "limitations": ["Previously examined historical data is not a fresh untouched holdout.",
                   "Costs use target-weight turnover, without holding drift, impact or execution constraints.",
                   "Source disclosure timing, revisions, survivorship and eligibility remain vendor-data checks.",
                   "A classification score trained on two tails is not a calibrated full-universe probability."]}
    (args.output/"summary.json").write_text(json.dumps(_json_safe(summary), indent=2, allow_nan=False)+"\n")
    print("SYNTHETIC DEMONSTRATION ONLY" if args.demo else "LOCAL DATA RESEARCH; inputs are not published")
    print(json.dumps(_json_safe({"output": str(args.output), "signal_metrics": signal_metrics,
                                 "selected_family": results["selection"]["family_winner"]}), indent=2))


if __name__ == "__main__":
    main()
