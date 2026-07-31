from __future__ import annotations
from typing import Sequence
from nee.experiments.runner import public_main

def main(argv: Sequence[str] | None = None) -> int:
    return public_main("exp04", argv)

if __name__ == "__main__":
    raise SystemExit(main())
