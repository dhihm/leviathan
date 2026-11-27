"""Build independent Metal Forge bundles in one Python process."""

import argparse
import json
import sys
import time

from dag_forge import ForgeRunError, build_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="trusted, deterministic local Python model")
    parser.add_argument("--example-input", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--helper")
    parser.add_argument("--job", nargs=2, action="append", required=True, metavar=("SEED", "OUT"))
    args = parser.parse_args(argv)
    try:
        jobs = [(int(seed), path) for seed, path in args.job]
    except ValueError:
        parser.error("--job seed must be an integer")
    for seed, path in jobs:
        started = time.perf_counter()
        try:
            report = build_bundle(args.model, args.example_input, args.input, path,
                                  seed=seed, helper=args.helper)
        except ForgeRunError as exc:
            print(f"dag_build_many: error: {path}: {exc}", file=sys.stderr)
            return 2
        print(json.dumps({"seed": seed, "buildMillis": (time.perf_counter() - started) * 1000,
                          "report": report}, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
