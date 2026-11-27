"""Run several inputs with one validated Metal Forge bundle and worker."""

import argparse
import json
import sys
from pathlib import Path

from dag_forge import ForgeRunError
from dag_runtime import LoadedRuntime


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--model", required=True, help="trusted local Python model file")
    parser.add_argument("--input", required=True, action="append", type=Path,
                        help="input JSON; repeat in execution order")
    parser.add_argument("--helper")
    args = parser.parse_args(argv)
    try:
        with LoadedRuntime(args.bundle, args.model, helper=args.helper) as runtime:
            startup = runtime.startupTimings
            results = []
            for path in args.input:
                try:
                    results.append(runtime.run(path))
                except ForgeRunError as exc:
                    raise ForgeRunError(f"input {path}: {exc}") from exc
        print(json.dumps({"startup": startup, "results": results}, indent=2, allow_nan=False))
    except ForgeRunError as exc:
        print(f"dag_run_many: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
