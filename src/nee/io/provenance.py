"""Runtime provenance containing only installed-package and official run data."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from nee import __version__


def _revision(root: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def runtime_provenance(root: str | Path) -> dict[str, Any]:
    commit, dirty = _revision(Path(root).resolve())
    return {
        "package": "numerical-einstein-equations",
        "package_version": __version__,
        "commit": commit,
        "dirty": dirty,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": importlib.metadata.version("matplotlib"),
        "pillow": importlib.metadata.version("pillow"),
        "platform": platform.platform(),
    }

