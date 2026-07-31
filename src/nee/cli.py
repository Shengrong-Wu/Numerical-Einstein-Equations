"""Command-line interface for validated experiment runs."""

from __future__ import annotations

import argparse
from typing import Sequence

from nee.experiments.registry import MODULES
from nee.experiments.runner import public_main


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nee")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("experiment", choices=sorted(MODULES))
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    return public_main(args.experiment, args.arguments)


if __name__ == "__main__":
    raise SystemExit(main())
