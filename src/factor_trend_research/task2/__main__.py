"""Run only an explicit generated demo or caller-supplied original inputs."""
import argparse
import json

from .demo import synthetic_inputs
from .inputs import load_native_inputs
from .runner import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="Generated synthetic software fixture; not market history")
    mode.add_argument("--native-inputs", help="Local prepared original-underlying directory with metadata/hashes")
    parser.add_argument("--output", required=True, help="Directory for CSV ledgers and labeled summary.json")
    parser.add_argument("--config", help="Optional task2 JSON rules and caps")
    parser.add_argument("--seed", type=int, default=20261114, help="Synthetic demo seed only")
    parser.add_argument("--demo-periods", type=int, default=520)
    args = parser.parse_args()
    data = synthetic_inputs(args.seed, args.demo_periods) if args.demo else load_native_inputs(args.native_inputs)
    _, summary = run(data, args.output, config_path=args.config)
    print(json.dumps({"mode": summary["mode"], "description": summary["description"],
                      "logical_assets": 16, "actual_legs": 20, "output": args.output,
                      "official_reference_replication_claim": False}, indent=2))


if __name__ == "__main__":
    main()
